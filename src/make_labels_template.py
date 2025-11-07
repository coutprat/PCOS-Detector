from pathlib import Path
import pandas as pd

DATA = Path(r"D:\PCOS\pcos-detection\data\processed")
manifest_csv = DATA / "manifest.csv"
ehr_parquet = DATA / "ehr.parquet"
out_template = DATA / "labels_template.csv"
out_labels = DATA / "labels.csv"

m = pd.read_csv(manifest_csv)
pids = pd.DataFrame(sorted(m["patient_id"].unique()), columns=["patient_id"])

if ehr_parquet.exists():
    ehr = pd.read_parquet(ehr_parquet)
    ehr.columns = [c.strip() for c in ehr.columns]
    label_cols = [c for c in ["pcos","PCOS","diagnosis","Diagnosis","label","Label","target","Target"] if c in ehr.columns]
    if label_cols:
        lab = ehr[["patient_id", label_cols[0]]].rename(columns={label_cols[0]:"label"})
        lab["label"] = (
            lab["label"].astype(str)
            .str.strip()
            .str.lower()
            .map({"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0,"pcos":1,"non-pcos":0})
        )
        labels = pids.merge(lab, on="patient_id", how="left")
        labels.to_csv(out_labels, index=False)
        print(f"[labels] Found column '{label_cols[0]}', wrote → {out_labels}")
    else:
        tmpl = pids.copy(); tmpl["label"] = ""
        tmpl.to_csv(out_template, index=False)
        print(f"[labels] No label column found in EHR. Fill this and rename to labels.csv → {out_template}")
else:
    tmpl = pids.copy(); tmpl["label"] = ""
    tmpl.to_csv(out_template, index=False)
    print(f"[labels] No ehr.parquet found. Fill this and rename to labels.csv → {out_template}")
