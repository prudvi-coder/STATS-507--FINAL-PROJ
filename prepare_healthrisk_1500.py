from datasets import load_dataset
import pandas as pd

def main():
    # 1) Load the dataset from Hugging Face
    ds = load_dataset("lvimuth/HealthRisk-1500-Medical-Risk-Prediction")  # [web:206]
    train_ds = ds["train"]

    print("Columns:", train_ds.column_names)

    # 2) Convert to DataFrame
    df = pd.DataFrame(train_ds)

    # 3) Select text + label columns
    text_col = "Doctor_Notes"
    label_col = "Risk_Level"

    df = df[[text_col, label_col]].rename(columns={text_col: "text"})

    # 4) Map labels to 0/1/2
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    df["risk_level"] = df[label_col].map(label_map)

    # 5) Keep only the columns your pipeline expects
    df = df[["text", "risk_level"]]

    # 6) Save to CSV
    out_path = "healthrisk_1500.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path} with columns: {df.columns.tolist()}")

if __name__ == "__main__":
    main()
