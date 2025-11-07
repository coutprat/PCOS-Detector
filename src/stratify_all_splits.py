from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROC = Path(r"D:\PCOS\pcos-detection\data\processed")  # change if needed
m = pd.read_csv(PROC/"manifest.csv")
l = pd.read_csv(PROC/"labels.csv")
l["label"] = l["label"].astype(int)

# one row per patient_id with its label
pids = l[["patient_id","label"]].drop_duplicates()

# 70/15/15 stratified split on patient_id
train_pids, temp_pids = train_test_split(
    pids, test_size=0.30, stratify=pids["label"], random_state=42
)
val_pids, test_pids = train_test_split(
    temp_pids, test_size=0.50, stratify=temp_pids["label"], random_state=42
)

def tag(df, subset, name):
    s = set(subset["patient_id"].tolist())
    df.loc[df["patient_id"].isin(s), "split"] = name

m["split"] = "train"
tag(m, val_pids,  "val")
tag(m, test_pids, "test")

# sanity: keep only known patients
known = set(pids["patient_id"])
m = m[m["patient_id"].isin(known)].reset_index(drop=True)

m.to_csv(PROC/"manifest.csv", index=False)

# report
merged = m.merge(l, on="patient_id", how="left")
print("by split & label:\n", merged.groupby(["split","label"]).size())
print("\nrows:", len(m))
