from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\PCOS\pcos-detection\data")
RAW = ROOT / "raw" / "ultrasound"
PROC = ROOT / "processed"

# 1) Build label map from RAW filenames (NEGATIVE-FIRST to handle 'noninfected')
pairs = []
for p in RAW.rglob("*"):
    if p.suffix.lower() not in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}:
        continue
    s = p.as_posix().lower()
    if any(k in s for k in ["noninfected","non-infected","negative","normal","control","healthy"]):
        lab = 0
    elif any(k in s for k in ["infected","pcos","positive","case","affected"]):
        lab = 1
    else:
        continue
    pairs.append((p.stem.lower(), lab))  # key by filename stem

raw_labels = pd.DataFrame(pairs, columns=["stem","label"]).drop_duplicates("stem")

# 2) Join with current manifest (patient_id == processed filename stem after rebuild)
man = pd.read_csv(PROC / "manifest.csv")
man["stem"] = man["image_path"].apply(lambda s: Path(s).stem.lower())
labels = (man[["patient_id","stem"]]
          .merge(raw_labels, on="stem", how="left")
          .dropna(subset=["label"])
          [["patient_id","label"]]
          .drop_duplicates("patient_id"))

out = PROC / "labels.csv"
labels.to_csv(out, index=False)

print(f"[labels] wrote -> {out}")
print("counts:\n", labels["label"].value_counts(dropna=False))
print("unique patient_ids:", labels["patient_id"].nunique())

# quick coverage check by split
merged = man.merge(labels, on="patient_id", how="inner")
print("by split (rows):\n", merged["split"].value_counts())
