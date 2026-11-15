# Reproducibility notes

## Current state

This repository contains the clinical NLP codebase, a copy of the HealthRisk-1500 CSV, and the submitted final report. The existing `main_pipeline.py` defaults to 500 samples and three classical plus three transformer model types; the final PDF reports a five-model comparison over 1,500 records.

**These two artifacts should not be represented as the same executed run.** The report-derived results are not asserted to have been reproduced from the committed default.

## To reproduce the report faithfully, recover

1. The final run's exact scripts or configuration, including model lists, `num_samples`, seeds, learning rates, epochs, oversampling decisions, and preprocessing.
2. The complete train/validation/test split (or code plus immutable source data/version identifying it).
3. Final stored model predictions, evaluation CSV, or logs from each model.
4. Precise dependencies/environment (and preferably commit hash) used for those runs.

Until then, the README keeps the report results distinctly attributed to the PDF. The existing code can be run as a **related implementation**, but its metrics must be independently measured.

## Re-run checklist

```bash
python prepare_healthrisk_1500.py
python main_pipeline.py
```

Run inside an isolated environment after installing `requirements.txt`. Validate input columns `text` and `risk_level`, review `CONFIG` before running, and report train/validation/test counts. Do not overwrite prior results if comparing runs.

## Responsible-use note

This is a course project illustrating model evaluation on a limited, de-identified text dataset, not a validated clinical decision-support tool.
