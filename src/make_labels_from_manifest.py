from pathlib import Path
import pandas as pd

PROC = Path(r"D:\PCOS\pcos-detection\data\processed")
man = pd.read_csv(PROC / "manifest.csv")

def infer_label_from_path(s: str):
    s = s.lower()
    # IMPORTANT: check negatives first so "noninfected" doesn't match "infected"
    if any(k in s for k in ["noninfected","non-infected","negative","normal","control","healthy"]):
        return 0
    if any(k in s for k in ["infected","pcos","positive","case","affected"]):
        return 1
    return None

# infer from image_path + patient_id text
tmp = man.assign(
    label=man["image_path"].fillna("").astype(str).apply(infer_label_from_path)
)

# if still None, try patient_id text as fallback
mask = tmp["label"].isna()
if mask.any():
    tmp.loc[mask, "label"] = tmp.loc[mask, "patient_id"].astype(str).apply(infer_label_from_path)

# drop rows with unknown label (should be none if folders are named clearly)
tmp = tmp.dropna(subset=["label"])

# if multiple rows per patient_id with conflicting labels, choose majority
labels = (tmp.groupby("patient_id")["label"]
          .agg(lambda x: int(round(x.mean())))   # majority vote
          .reset_index())

out = PROC / "labels.csv"
labels.to_csv(out, index=False)
print(f"[labels] wrote -> {out}")
print("counts:\n", labels["label"].value_counts(dropna=False))
print("unique patient_ids:", labels["patient_id"].nunique())

# sanity: show coverage by split after join
merged = man.merge(labels, on="patient_id", how="inner")
print("by split (rows):\n", merged["split"].value_counts())
