from pathlib import Path
import pandas as pd

# Adjust if your processed path is different
PROC = Path(r"D:\PCOS\pcos-detection\data\processed")

manifest = pd.read_csv(PROC / "manifest.csv")

def infer_label(row):
    s = (str(row.get("patient_id","")) + " " + str(row.get("image_path",""))).lower()
    # Positive class keywords (PCOS present)
    if any(k in s for k in ["infected", "pcos", "positive", "case", "affected"]):
        return 1
    # Negative class keywords (PCOS absent)
    if any(k in s for k in ["noninfected", "non-infected", "negative", "normal", "control", "healthy"]):
        return 0
    return None

labels = (manifest
          .assign(label=manifest.apply(infer_label, axis=1))
          [["patient_id","label"]]
          .dropna()
          .drop_duplicates("patient_id"))

out = PROC / "labels.csv"
labels.to_csv(out, index=False)

print(f"[labels] wrote → {out}")
print(f"unique patient_ids: {labels['patient_id'].nunique()}  |  counts:\n{labels['label'].value_counts(dropna=False)}")
