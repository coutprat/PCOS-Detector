import argparse, os, re, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def deterministic_split(key: str, p_train=0.7, p_val=0.15):
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) / (1<<128)
    return "train" if h < p_train else ("val" if h < p_train + p_val else "test")

def load_ehr(ehr_dir: Path) -> pd.DataFrame:
    files = sorted(list(ehr_dir.glob("*.csv")) + list(ehr_dir.glob("*.xlsx")) + list(ehr_dir.glob("*.xls")))
    if not files: return pd.DataFrame()
    dfs = []
    for f in files:
        df = pd.read_csv(f) if f.suffix.lower()==".csv" else pd.read_excel(f)
        df["__source_file"] = f.name
        dfs.append(df)
    ehr = pd.concat(dfs, ignore_index=True)
    ehr.columns = [c.strip() for c in ehr.columns]
    cand = [c for c in ehr.columns if c.lower() in {"patient_id","pid","patientid","id"}]
    ehr.rename(columns={cand[0]:"patient_id"} if cand else {}, inplace=True)
    if "patient_id" not in ehr.columns:
        ehr["patient_id"] = ehr.index.map(lambda i: f"pt_{i:05d}")
    ehr = ehr.replace(["NA","NaN","nan",""], np.nan).drop_duplicates(subset=["patient_id"], keep="first")
    return ehr

def normalise_save_image(src: Path, dst: Path, size=448):
    img = Image.open(src).convert("L")
    w,h = img.size; m = min(w,h)
    left=(w-m)//2; top=(h-m)//2
    img = img.crop((left, top, left+m, top+m)).resize((size,size))
    ensure_dir(dst.parent); img.save(dst, format="PNG", optimize=True)

def main(data_root: Path):
    raw_ultra = data_root / "raw" / "ultrasound"
    raw_ehr   = data_root / "raw" / "ehr"
    proc_dir  = data_root / "processed"
    out_img   = proc_dir / "images"
    ensure_dir(proc_dir); ensure_dir(out_img)

    ehr = load_ehr(raw_ehr)
    if not ehr.empty: ehr.to_parquet(proc_dir / "ehr.parquet", index=False)
    else: print("WARN: No EHR files found; continuing with images only.")

    IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".dcm"}
    files = [p for p in raw_ultra.rglob("*") if p.suffix.lower() in IMG_EXTS]
    rows = []
    pid_pat = re.compile(r"(?:^|[_-])(pt\d+|patient\d+|id\d+)", re.I)

    for src in sorted(files):
        name = src.stem
        m = pid_pat.search(name)
        pid = (m.group(1).lower() if m else src.parent.name.lower())
        split = deterministic_split(pid)
        dst = out_img / split / f"{name}.png"
        normalise_save_image(src, dst)
        rows.append({"image_path": str(dst.relative_to(proc_dir)), "patient_id": pid, "split": split})

    manifest = pd.DataFrame(rows)
    if not manifest.empty: manifest.to_csv(proc_dir / "manifest.csv", index=False)
    print(f"Images: {len(manifest)} | EHR rows: {0 if ehr.empty else len(ehr)}")
    print(f"Wrote: {proc_dir/'manifest.csv'}  {proc_dir/'ehr.parquet'} (if EHR present)")
    print(rf"Images saved under: {out_img}\(train|val|test)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    default_data = str(Path(__file__).resolve().parents[1] / "data")
    ap.add_argument("--data-root", default=default_data, help=r'e.g., D:\PCOS\pcos-detection\data (defaults to project/data)')
    args = ap.parse_args()
    main(Path(args.data_root))
