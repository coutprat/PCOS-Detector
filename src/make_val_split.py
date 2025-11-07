from pathlib import Path
import pandas as pd
import hashlib

PROC = Path(r"D:\PCOS\pcos-detection\data\processed")
m = pd.read_csv(PROC / "manifest.csv")
labs = pd.read_csv(PROC / "labels.csv")

# Keep only rows that have labels
m = m.merge(labs, on="patient_id", how="inner")

# If there's already some 'val', keep them; otherwise create ~15% val from current train
def hash01(s: str) -> float:
    h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
    return h / float(1 << 128)

has_val = (m["split"] == "val").any()
if not has_val:
    # preserve test rows; resplit only the non-test rows into train/val deterministically
    non_test = m["split"] != "test"
    key = m.loc[non_test, "patient_id"].astype(str)
    r = key.apply(hash01)
    # 15% to val, rest stay train
    m.loc[non_test & (r < 0.15), "split"] = "val"
    m.loc[non_test & (r >= 0.15), "split"] = "train"

# Drop the label column we added (train script merges labels itself)
m = m.drop(columns=["label"])

# Write back
out = PROC / "manifest.csv"
m.to_csv(out, index=False)

# Show counts
print("by split (rows):")
print(m["split"].value_counts())
