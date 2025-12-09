# Clinical Risk Classification from Free-Text Notes

This project builds an end-to-end machine learning pipeline to predict patient risk level (Low / Medium / High) from clinical notes.

## Dataset

- Source: HealthRisk-1500 (Hugging Face)
- 1,500 de-identified patient records
- Each record includes symptoms, doctor notes, and a three-class risk label (Low / Medium / High)

## How to Prepare the Data

Run this script once to download and create the CSV:
python prepare_healthrisk_1500.py

This creates `healthrisk_1500.csv` in your project folder with:
- `text` (doctor notes)
- `risk_level` (0 = Low, 1 = Medium, 2 = High)

## How to Run the Pipeline

1. Create a virtual environment (first time only):
    ```
    python -m venv venv
    ```
2. Activate the environment:
    ```
    venv\Scripts\activate
    ```
3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
4. Run the ML pipeline:
    ```
    python main_pipeline.py
    ```

## Models Included

- Logistic Regression (LR)
- Random Forest (RF)
- XGBoost (XGB)
- BioBERT
- PubMedBERT

## Outputs (in `./results/`)

- Metrics summary (CSV / console)
- `model_comparison.png`
- `per_class_metrics.png`
- Confusion matrices for each model
- Error analysis CSVs (e.g. `errors_LR.csv`)

## Notes

- Random seeds are set in `utils.py` via `set_seed(42)`.
- The project uses a single stratified train/val/test split with oversampling on the train set.
- Cross-validation and extensive hyperparameter tuning are left as future work.