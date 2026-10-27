# gnn-bert-music-context

CSE425/EEE474/CSE715 Neural Networks final project: GNN--BERT fusion for
music context understanding. Built against **LP-MusicCaps-MTT** (real
captions over MagnaTagATune audio) -- see `report/DECISIONS.md` for the
verified distinction between full-split baselines and neural pilot runs.

## tl;dr status

| Layer | Status |
|---|---|
| `src/captions_data.py` (parse captions/tags/paths) | **real, tested** (`tests/test_captions_data.py`) |
| `src/graph_builder.py` / `audio_features.py` (graph construction logic) | **real, tested** on synthetic tensors AND a genuine audio fixture (`tests/test_graph_builder.py`, `tests/test_audio_pipeline_synthetic.py`) |
| `src/baselines.py` B1 majority-class / random | **real numbers**, run on the actual validation split (`results/metrics.json`) |
| `src/graph_dataset.py`, `bert_encoder.py`, `gnn_model.py`, `fusion_model.py`, `contrastive.py` | **implemented and tested**: direct-CLS BERT head, GraphSAGE/GAT, cross-attention and early-concat fusion, symmetric InfoNCE with learnable temperature |
| `src/train.py --task {1,2,3,4}` | **complete runnable workflows** with checkpoints, learning curves, Task-1 attention cases, Task-2 CNN comparison, Task-3 ablations, and Task-4 bidirectional retrieval |
| `report/final_report.tex` | **compiled** to `final_report.pdf` with dataset statistics, pilot results, curves, attention cases, ablations, and limitations |

30/30 tests passing (`pytest tests/`). See `report/DECISIONS.md` Section 5
for exactly what's real vs. mocked and why.

## Setup

```bash
pip install -r requirements.txt
```

## Data

The LP-MusicCaps-MTT metadata and local MagnaTagATune audio used for this
project are included under `data/`, along with processed features and model
checkpoints. See the [dataset card](data/lp_musiccaps/README.md) for dataset
details and `report/DECISIONS.md` for the experiment files and split
generation details.

To download the audio needed for Tasks 2-4:

```bash
bash scripts/download_audio.sh          # real MagnaTagATune URLs, ~3GB, needs internet
```

An older MTG-Jamendo-based path (`src/data.py`) is also included and tested,
kept as a documented secondary option -- see `report/DECISIONS.md` Section 4
for why it's not primary. Its metadata and split files are not included.

## Run

```bash
python src/train.py --task 1 --config config.yaml # BERT + curves + 5 attention examples
bash scripts/download_audio.sh                    # prerequisite for Tasks 2-4 (~3 GB)
python src/train.py --task 2 --config config.yaml # GraphSAGE/GAT versus 2D CNN
python src/train.py --task 3 --config config.yaml # BERT/GNN/early-concat/cross-attention ablation
python src/train.py --task 3 --emotion            # uses valence/arousal only when those columns exist
python src/train.py --task 4 --config config.yaml # InfoNCE + test R@1/R@5/R@10 both directions
python -m pytest -q                               # 30 tests
```

## Report

```bash
cd report && pdflatex final_report.tex && bibtex final_report && pdflatex final_report.tex && pdflatex final_report.tex
```
Already compiled: `report/final_report.pdf`.

## Repo layout

```
gnn-bert-music-context/
  data/
    lp_musiccaps/             # LP-MusicCaps-MTT metadata
    raw/audio/                # downloaded MagnaTagATune audio
    processed/                # cached audio features
  src/
    config.py                # central config (dataclasses) + config.yaml loader
    captions_data.py          # PRIMARY: LP-MusicCaps-MTT loader (real, tested)
    data.py                    # legacy: MTG-Jamendo loader (real, tested)
    audio_features.py          # librosa mel/MFCC/chroma segmentation
    graph_builder.py            # chord-transition + segment-similarity graphs (real, tested)
    graph_dataset.py             # audio -> torch_geometric.data.Data glue (Tasks 2-4)
    bert_encoder.py               # Task 1 BERT tag classifier
    gnn_model.py                   # Task 2 GraphSAGE/GAT encoder
    fusion_model.py                 # Task 3 cross-attention fusion
    contrastive.py                   # Task 4 InfoNCE dual encoder
    baselines.py                      # B1-B4 baselines
    evaluate.py                        # macro/micro F1, mean AP, R@k, emotion MAE/R2
    train.py                            # CLI: --task {1,2,3,4}
    utils.py
  scripts/
    download_audio.sh                    # real MagnaTagATune URLs (see DECISIONS.md)
    mtgjamendo_scripts/                   # upstream MTG-Jamendo repo scripts (legacy path)
  tests/                                   # 30 passing model/data/audio/integration tests
  results/
    *.json, *.png                           # metrics, histories, and figures
    *.pt                                    # model checkpoints
  report/
    final_report.tex, final_report.pdf, neurips_2026.sty, references.bib
    DECISIONS.md                            # what's real vs. simulated, and why -- read this first
  config.yaml
  requirements.txt
```

## Read this first if something looks off

`report/DECISIONS.md` documents the dataset choice, the exact pilot sizes,
what was executed with real pretrained weights/audio, and why pilot metrics
must not be presented as full-corpus benchmark results.
