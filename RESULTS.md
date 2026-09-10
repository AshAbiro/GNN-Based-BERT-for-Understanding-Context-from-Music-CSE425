# MusicCaps Multimodal Experiments

All reported numbers use the held-out test split. Model selection used validation data only.

## Data

- Source rows/audio files: 761 paired MusicCaps examples.
- Prepared examples: 760 (one row had no tag surviving the frequency filter).
- Targets: 124 AudioSet-label proxies with minimum corpus frequency 5.
- Iterative multi-label split: 532 train / 114 validation / 114 test.
- Seed: 42.

## Task 1 — BERT multi-label classifier

DistilBERT `[CLS]` hidden state followed directly by a linear `W, b` head, trained for 8 epochs with `BCEWithLogitsLoss`.

| Test Macro-F1 | Test Micro-F1 |
|---:|---:|
| 0.1622 | 0.3392 |

Artifacts: `artifacts/task1_bert/`. This includes the checkpoint, epoch-level F1 curve, prediction JSON, and five last-layer `[CLS]` attention visualizations.

## Task 2 — structure graph versus mel CNN

Graph nodes are temporal audio segments represented by MFCC/chroma features. GraphSAGE uses mean aggregation and global mean pooling. Both models trained for 30 epochs.

| Model | Best validation Macro-F1 | Test Macro-F1 | Test Micro-F1 |
|---|---:|---:|---:|
| GraphSAGE | 0.1522 | 0.1006 | 0.2936 |
| 2D mel CNN | 0.1175 | 0.0873 | 0.2033 |

GraphSAGE improved test Macro-F1 by 0.0133 and Micro-F1 by 0.0903 over the mel baseline.

Artifacts: `artifacts/task2_gnn/`.

## Task 3 — multimodal ablation

Each variant trained for an equal 4-epoch budget. Cross-attention uses the graph readout as query and DistilBERT token states as keys/values.

| Variant | Best validation Macro-F1 | Test Macro-F1 | Test Micro-F1 |
|---|---:|---:|---:|
| Cross-attention fusion | 0.0620 | 0.0685 | 0.2628 |
| BERT-only | 0.1068 | 0.1096 | 0.2873 |
| GNN-only | 0.0278 | 0.0375 | 0.2555 |
| Early concatenation | 0.1818 | 0.1770 | 0.3361 |

The supplied MusicCaps table has no DEAM valence/arousal targets. The implementation supports masked auxiliary MSE losses, but those branches were inactive here; recorded MSE values are null.

Artifacts: `artifacts/task3_fusion/`.

## Task 4 — cross-modal contrastive retrieval

The dual encoder was trained for 20 epochs with symmetric InfoNCE and a learnable temperature. Epoch 19 was selected by the mean of the six validation recalls.

| Direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Audio graph to caption | 0.0614 | 0.2368 | 0.3772 |
| Caption to audio graph | 0.0789 | 0.2193 | 0.3860 |

Artifacts: `artifacts/task4_contrastive/`.

## Interpretation and limitations

- Early concatenation is the strongest four-epoch fusion variant on this small split. Cross-attention likely needs more data or tuning; this run does not establish a general architectural ranking.
- Tag Macro-F1 is suppressed by the 124-label long tail, while Micro-F1 emphasizes common labels.
- Retrieval uses exact paired-item relevance. Semantically valid alternative captions are counted as misses.
- Results are a single-seed experiment and should be repeated with multiple seeds before drawing statistical conclusions.
- The decision threshold is fixed at 0.5; per-label validation calibration could improve multi-label F1.

## Verification

`python -B -m unittest discover -s tests -v` passes all 6 tests.
