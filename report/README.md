# Final project report

This directory contains the current report files for **GNN-Based BERT for Understanding Context from Music (CSE425)**:

- [Current PDF](GNN_Based_BERT_for_Understanding_Context_from_Music.pdf)
- [Edited LaTeX source](GNN_BERT_Final_Report_Replaced_Texts.tex)
- [Edited LaTeX source with clickable citations](GNN_BERT_Final_Report_Clickable_Citations.tex)

The current PDF has 11 pages. Both edited sources were copied unchanged from the author's open files. The PDF is included as supplied; it was not regenerated from these sources during the upload.

The edited sources contain references to VQ-VAE/image encoding, video translation, and changed split/seed settings that are not supported by the implemented experiments. The verified experimental record is [RESULTS.md](../RESULTS.md) and the saved files under [artifacts](../artifacts/). These discrepancies should be corrected before treating the edited prose as a verified experimental description.

The manuscript uses a generic two-column conference layout; it is not an official NeurIPS, IEEE, or ICML submission template. Transfer the content to a venue-specific template if one is required by the course.

## Build

From this directory, use either:

```powershell
tectonic --keep-logs GNN_BERT_Final_Report_Clickable_Citations.tex
```

or run `pdflatex GNN_BERT_Final_Report_Clickable_Citations.tex` twice in a normal LaTeX installation. Upload that source and the `figures` directory together to Overleaf. No separate BibTeX step is needed. A LaTeX installation is required separately; compiler binaries are not committed.

## Figures and provenance

Seven figures are supplied as vector PDFs and PNG previews. Figures 1, 2, 4, and 6 are schematics of the inspected implementation. They do not represent new measurements. Figures 3, 5, and 7 are derived only from saved experimental outputs:

| Figure | Source |
|---|---|
| 1: system overview | `src/musiccaps_ml/models.py` and training scripts |
| 2: graph pipeline | `src/musiccaps_ml/audio.py` |
| 3: BERT learning curves | `artifacts/task1_bert/learning_metrics.csv` |
| 4: fusion | `MultimodalAblationModel` in `models.py` |
| 5: ablation comparison | `artifacts/task3_fusion/ablation_results.csv` |
| 6: dual encoder | `GraphTextDualEncoder` and `symmetric_infonce` in `models.py` |
| 7: attention sample | First entry in `artifacts/task1_bert/prediction_samples.json` |

From the project root, regenerate figures with:

```powershell
python report/build_report_assets.py
```

The numerical results were reconciled with saved configurations and metric files under `artifacts/task1_bert`, `artifacts/task2_gnn`, `artifacts/task3_fusion`, and `artifacts/task4_contrastive`. Classification metrics are reported independently of retrieval metrics. The full split was used; smoke-test results are excluded.

## Evidence corrections applied

- The standalone GraphSAGE encoder uses normalization and ReLU, but no dropout. Dropout is present in the common Task 3 prediction head.
- Task 3's BERT-only model differs from Task 1 in prediction head and optimizer setup, as well as epochs and batch size. Cross-task results are not a controlled head-for-head comparison.
- Audio similarity edges are recurrence proxies, not annotated musical-form relationships.
- The vocabulary frequency filter was applied before splitting. The report states this conditioning explicitly.
- Valence/arousal heads and masked losses were implemented, but no emotion-supervised training/evaluation occurred in the reported runs.
- The retrieval-order JSON contains test item IDs, not a documented set of qualitative ranking case studies.
- Python 3.11–3.14 is the package's declared compatibility range, not evidence of a complete interpreter test matrix.
- The original attention values are diagnostic, class-agnostic weights, not validated causal explanations.

## References and submission details

The original report's foundational references were checked against primary paper pages on ACL Anthology and arXiv. The supplied edited versions add a MusicCaps dataset reference and retain the embedded bibliography.

The edited sources include the authors, student IDs, and BRAC University affiliation. Embedding visualization, qualitative retrieval cases, curated graph exports, and a demonstration notebook were not confirmed as completed deliverables during the original experiment audit.
