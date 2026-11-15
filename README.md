# Clinical Risk Classification from Free-Text Notes

A university Data Science project comparing classical TF-IDF classifiers against biomedical transformer models for **three-class prediction** (Low, Moderate, High) from short text notes. The purpose is to understand **model selection, evaluation, and errors under limited labeled data**, not to offer clinical triage.

> **Research/coursework only. Not intended for patient care, diagnosis, or automated clinical decisions.** Reported performance is too limited for real-world triage.

## Highlights from the submitted project report

- **Data described in the report:** 1,500 HealthRisk-1500 text notes (three balanced classes)
- **Methods:** TF-IDF + logistic regression / random forest / XGBoost; fine-tuned BioBERT and PubMedBERT
- **Evaluation:** stratified train/validation/test split, macro F1, precision, recall and class-specific error analysis
- **Reported findings:** Random Forest macro F1 **0.39**, logistic regression and XGBoost **0.38**, BioBERT **0.34**, PubMedBERT **0.31**.
- **Interpretation:** on this dataset, simpler baselines were more competitive than larger models; the Moderate class was particularly challenging.

These results are transcribed from the submitted `Final REPORT (1).pdf`, not freshly recomputed from this GitHub checkout.

## IMPORTANT: report vs. current code configuration

**The code as currently committed does not reproduce the report's exact experiment out of the box.** In `main_pipeline.py`:

- `CONFIG['data']['num_samples']` is **500**, whereas the report describes **1,500** notes.
- The current transformer list includes **ClinicalBERT** as a sixth model, while the report's performance table includes five models.
- `balance=False` in the entry point, but the older README claims oversampling. The submitted report describes specific training refinements and configurations not captured by a single committed default.
- Model hyperparameters may differ from the final tuned settings used for the report.

The report and current code should be treated as **related project artifacts, not a guaranteed bit-for-bit reproduction**. Do not quote the reported macro F1 values as if they came from the default 500-row run. See `docs/REPRODUCIBILITY.md`.

## Repository contents

| Path | Role |
| --- | --- |
| `prepare_healthrisk_1500.py` | Downloads/transforms the HealthRisk-1500 dataset |
| `data_prep.py` | Text normalization, label stratification, optional oversampling, PyTorch datasets |
| `models.py` | TF-IDF baselines and transformer fine-tuning |
| `main_pipeline.py` | Configurable training/evaluation entry point |
| `evaluate_explain.py` | Scores, confusion matrices, per-class metrics and error analysis |
| `Final REPORT (1).pdf` | Submitted project methodology, results and limitations |

## Quick start (current code configuration)

Create a virtual environment, install requirements, then prepare the dataset and run the pipeline:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python prepare_healthrisk_1500.py
python main_pipeline.py
```

The current default loads up to **500** observations from `healthrisk_1500.csv` and runs on CPU unless CUDA is available. Transformer training can be computationally expensive. Review the settings in `main_pipeline.py` before starting.

**To explore a larger dataset**, change the `num_samples` setting explicitly, but do not assume the report's exact metrics will be reproduced. The complete experimental configuration and saved runs must first be reconciled.

## Limitations and ethics

- Single holdout split with limited hyperparameter search; no demonstrated external validation.
- Text-only features omit critical clinical information and may cause unsafe errors.
- Class-wise errors, particularly under-triage of High-risk examples, are more important than overall accuracy for any potential medical application.
- Use only appropriately licensed/de-identified data; review the dataset's redistribution terms before publishing copies.

## Future work

Reconstruct the exact final experiment configuration; add stable config files and pinned requirements; commit aggregate metrics and plots; add lightweight unit tests; and, separately, validate on independent datasets before drawing generalized conclusions.
