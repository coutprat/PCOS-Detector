from pathlib import Path
import pandas as pd
PROC = Path(r"D:\PCOS\pcos-detection\data\processed")
m = pd.read_csv(PROC/"manifest.csv")
l = pd.read_csv(PROC/"labels.csv")
df = m.merge(l, on="patient_id", how="inner")
print(df.groupby(["split","label"]).size().unstack(fill_value=0))
