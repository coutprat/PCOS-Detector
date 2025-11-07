from pathlib import Path
import pandas as pd

PROC = Path(r"D:\PCOS\pcos-detection\data\processed")
images = PROC / "images"  # images/<split>/*.png

rows = []
for split_dir in images.iterdir():
    if not split_dir.is_dir(): 
        continue
    split = split_dir.name  # train / val / test
    for p in split_dir.glob("*.png"):
        rows.append({
            "image_path": str(p.relative_to(PROC)).replace("\\","/"),
            "patient_id": p.stem.lower(),   # unique per image
            "split": split
        })

df = pd.DataFrame(rows).sort_values("image_path").reset_index(drop=True)
out = PROC / "manifest.csv"
df.to_csv(out, index=False)
print(f"[manifest] wrote -> {out}  rows={len(df)}  unique patient_ids={df['patient_id'].nunique()}")
