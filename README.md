# GNN-Based BERT for Understanding Context from Music (CSE425)

Neural Networks course project combining DistilBERT, GraphSAGE, multimodal fusion,
and contrastive audio-caption retrieval on a MusicCaps subset.

- [Current project report (PDF)](report/GNN_Based_BERT_for_Understanding_Context_from_Music.pdf)
- [Report sources and provenance notes](report/README.md)
- [Experimental results](RESULTS.md)
- [Saved evaluation artifacts](artifacts/)

The prepared dataset contains 760 usable examples, 124 tags, and a
532/114/114 train/validation/test split. Early concatenation achieved test
Macro-/Micro-F1 of 0.1770/0.3361 in the four-epoch fusion ablation. The separate
eight-epoch DistilBERT classifier achieved 0.1622/0.3392. These are single-seed
results with different cross-task training budgets.

This package implements four experiments over the local MusicCaps subset:

1. BERT `[CLS]` multi-label classification with BCE loss, Macro/Micro-F1 curves,
   and five last-layer attention visualizations.
2. GraphSAGE over temporal MFCC/chroma music graphs, compared with a 2D mel-CNN.
3. GNN-to-BERT cross-attention fusion with BERT-only, GNN-only, and early-fusion
   ablations. Optional valence/arousal regression is enabled when target columns exist.
4. Symmetric InfoNCE alignment of graph and caption embeddings, evaluated with
   bidirectional R@1, R@5, and R@10.

The CSV's obsolete Colab paths are deliberately ignored; audio is resolved as
`audio_clips/<ytid>.wav`. All tasks use the same persisted stratified split.

## Setup

Use Python 3.11–3.14 with package versions satisfying `pyproject.toml`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For CUDA, install the correct PyTorch build for the machine first, then install this
project. PyTorch Geometric wheels must match that PyTorch/CUDA combination.

## Prepare the shared experiment data

Obtain the MusicCaps metadata and corresponding audio separately. Place the input
table at `musiccaps_with_audio_paths.csv` and WAV files at `audio_clips/<ytid>.wav`,
or supply `--csv` and `--audio-dir` explicitly. Audio, raw metadata, caches, and
model checkpoints are not included in this repository.

The original prepared split is recorded in
[`artifacts/data/split_assignments.csv`](artifacts/data/split_assignments.csv),
with the vocabulary and split settings alongside it. These small files record
the experiment; rerun preparation with the same source subset to generate the
local `dataset.csv` consumed by the training scripts.

```powershell
python scripts/prepare_data.py
```

Defaults use AudioSet IDs from `audioset_positive_labels`, retaining tags appearing
at least five times. Use `--target-source aspects` only if caption-derived proxy tags
are intended; doing so makes text-to-tag prediction partially tautological.

## Run tasks

```powershell
# Task 1
python scripts/train_bert.py --model-name distilbert-base-uncased --epochs 8

# Task 2: GraphSAGE and mel-CNN baseline
python scripts/train_gnn.py --epochs 30

# Task 3: all four ablations
python scripts/train_fusion.py --epochs 4 --variants fusion bert gnn early

# Task 4
python scripts/train_contrastive.py --epochs 20
```

Use `--help` on each command for batch sizes, learning rates, cache locations, and
other settings. `--limit-per-split` is available for fast integration checks and must
not be used for reported experiments. The graph/fusion tasks cache extracted graph
features under `cache/`.

## Outputs

Each run creates a timestamp-free, task-specific directory under `artifacts/` with:

- model checkpoint(s);
- epoch metrics in CSV;
- Macro-F1 and Micro-F1 learning-curve PNGs;
- configuration and vocabulary metadata;
- attention plots and sample prediction JSON for Task 1;
- ablation comparison CSV for Task 3;
- bidirectional retrieval metrics for Task 4.

## Important experimental notes

- The dataset has 761 examples, 229 raw AudioSet labels, and substantial imbalance.
  Report both Macro-F1 and Micro-F1 and preserve the generated split manifest.
- The included deterministic iterative multilabel splitter attempts to preserve rare
  label proportions while maintaining exact train/validation/test sizes.
- Five clips are mono and the remainder stereo; sample rates are 44.1 or 48 kHz.
  Audio loading always converts to mono, resamples, and pads/crops to a fixed length.
- The current CSV has no DEAM valence/arousal values. Task 3 therefore optimizes tag
  loss only and records the auxiliary losses as unavailable. Add numeric `valence`
  and `arousal` columns to enable the full objective.
- Attention weights are diagnostic, not guaranteed causal explanations.
- Audio licensing/provenance must be reviewed before redistributing the WAV files.
