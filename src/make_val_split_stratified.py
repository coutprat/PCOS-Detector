from pathlib import Path
import pandas as pd
import hashlib

PROC = Path(r"D:\PCOS\pcos-detection\data\processed")
m = pd.read_csv(PROC/"manifest.csv")
l = pd.read_csv(PROC/"labels.csv")
df = m.merge(l, on="patient_id", how="inner")

# keep test as-is; resplit only non-test into train/val stratified by label
non_test = df["split"] != "test"
df_nt = df.loc[non_test].copy()

def hash01(s: str) -> float:
    h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
    return h / float(1 << 128)

# per-class hashing for ~15% val
for lab in [0,1]:
    mask = df_nt["label"] == lab
    r = df_nt.loc[mask, "patient_id"].astype(str).apply(hash01)
    df_nt.loc[mask & (r < 0.15), "split"] = "val"
    df_nt.loc[mask & (r >= 0.15), "split"] = "train"

# write back into original manifest layout
df_out = df.copy()
df_out.loc[non_test, "split"] = df_nt["split"]

# drop helper label column
df_out = df_out.drop(columns=["label"])
df_out.to_csv(PROC/"manifest.csv", index=False)

# show final balance
merged = m.merge(l, on="patient_id", how="inner")
merged["split"] = df_out["split"]  # updated
print(merged.groupby(["split","label"]).size().unstack(fill_value=0))
