# Verified project state and design decisions

This file records what is implemented, what was executed, and which numbers
are pilots rather than publishable full-corpus benchmarks.

## 1. Primary dataset

The primary dataset is LP-MusicCaps-MTT: LLM-generated natural-language
captions paired with MagnaTagATune metadata and audio paths. The target is the
50-tag multi-hot vector and the default text field is `caption_writing`.
This directly supports caption-to-tag classification and caption/audio
alignment. The older MTG-Jamendo path remains as a legacy secondary dataset;
it must not be mixed with LP-MusicCaps because their track identities differ.

Verified unfiltered rows are 18,706 train, 1,825 validation, and 5,329 test.
After requiring a non-empty caption and at least one tag, 15,247 / 1,529 /
4,332 rows remain. The 50- and 188-tag vocabularies have their advertised
sizes. `caption_attribute_prediction` and `pseudo_attribute` are deliberately
excluded because the dataset card warns against those fields.

## 2. Audio and dependencies

The official MagnaTagATune server was unavailable during setup, so
`scripts/download_audio.sh` now falls back to the consolidated archive used by
the public `confit/magnatagatune` dataset loader. The local archive contains
25,863 MP3 files. All 25,860 paths referenced by the three LP-MusicCaps
metadata files resolve under `data/raw/audio`.

The active environment has PyTorch, Transformers, PyTorch Geometric, librosa,
PyArrow, and pytest. Pretrained `distilbert-base-uncased` weights are cached
and were used in the pilot runs. The machine has a four-core Intel i5-11300H
and no CUDA device.

## 3. Implemented tasks

- Task 1 uses one linear classification head directly on the first/CLS token,
  BCE-with-logits loss, per-epoch Macro-F1/Micro-F1/mean AP, test evaluation,
  a learning-curve figure, and five prediction/attention examples. Eager
  attention is selected because the SDPA backend does not return attention
  tensors.
- Task 2 builds five-second music-structure graphs with 20 MFCC and 12 chroma
  node features, temporal/similarity edges, GraphSAGE or GAT, global mean
  pooling, and a 2D log-mel CNN comparison.
- Task 3 implements graph-query/text-key-value cross-attention, optional
  valence/arousal MSE terms, and BERT-only, GNN-only, early-concatenation, and
  cross-attention ablations. LP-MusicCaps has no valence/arousal columns, so
  the auxiliary emotion objective is correctly disabled for this dataset.
- Task 4 implements normalized graph/text dual encoders, symmetric InfoNCE, a
  learnable positive temperature, and bidirectional R@1/R@5/R@10 on test data.

Each workflow saves JSON history/results and model checkpoints under
`results/`. Tasks 1--3 also save the required plots.

## 4. Executed results and interpretation

Full-split validation baselines use 15,247 training and 1,529 validation
examples. The majority predictor obtains Macro/Micro-F1 0/0 and mean AP
0.066; the seeded random predictor obtains 0.108/0.116 and mean AP 0.069.

Neural runs are explicitly labeled pilots in `results/metrics.json`:

- Task 1: 256 examples per split, two epochs, genuine pretrained DistilBERT.
- Tasks 2--4: 12 examples per required split, one epoch, genuine audio.

These pilots verify data loading, optimization, evaluation, serialization,
plots, attention extraction, ablation orchestration, and retrieval metrics.
They are far too small and short to rank models or establish generalization.
For example, Task 4 recall equals the chance values for a 12-item retrieval
set. Full-corpus, multi-seed GPU runs are still required for scientific
claims. Never copy pilot values into a table without retaining the `pilot`
label and sample sizes.

## 5. Verification

`python -m pytest -q` passes 30/30 tests. Coverage includes real parquet
loading and split isolation, graph construction, MFCC/chroma extraction from
a known A/B/A waveform, GraphSAGE/GAT forward passes, direct-CLS BERT
classification, cross-attention masking, early concatenation, multi-task
loss, learnable-temperature gradients, InfoNCE and recall, graph/text
collation, and a complete Task 1 loop. Transformer unit tests use a tiny local
model for speed; the saved pilots use actual pretrained weights.

The final report is compiled at `report/final_report.pdf`. Its results section
separates full-split baselines from pilot neural measurements and includes the
Task 1 curve and attention panel plus the Task 3 ablation figure.

## 6. Reproduction commands

```bash
pip install -r requirements.txt
python -m pytest -q
python src/train.py --task 1 --config config.yaml
python src/train.py --task 2 --config config.yaml
python src/train.py --task 3 --config config.yaml
python src/train.py --task 4 --config config.yaml
```

Omitting `--max-audio-samples` uses every available row. For a clearly
labeled development pilot, add `--max-audio-samples N`; despite its historical
name, this limit applies to Task 1 as well.
