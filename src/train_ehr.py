import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Target feature set (matches the React ehrSchema.ts)
TARGET_FEATURES = [
    "age", "bmi",
    "amh", "total_testosterone",
    "hirsutism", "acne_severity",
    "cycle_regularity", "avg_cycle_length_days",
    "fasting_insulin", "fasting_glucose",
]

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lowmap = {c: c.lower().strip().replace(" ", "_") for c in df.columns}
    inv = {v: k for k, v in lowmap.items()}

    def has(key): return key in inv
    def get_num(key, default=np.nan):
        return pd.to_numeric(df[inv[key]], errors="coerce") if has(key) else pd.Series(default, index=df.index)
    def get_cat_yn(key, yes=1, no=0, default=np.nan):
        if not has(key): return pd.Series(default, index=df.index)
        s = df[inv[key]].astype(str).str.strip().str.lower()
        return s.map({"y": yes, "yes": yes, "n": no, "no": no, "1": yes, "0": no}).fillna(default)

    # label from PCOS (Y/N)
    if "label" not in df.columns:
        pcos_keys = ["pcos_(y/n)", "pcos_y/n", "pcos"]
        lbl = None
        for k in pcos_keys:
            if has(k):
                s = df[inv[k]].astype(str).str.strip()
                lbl = s.map({"1": 1, "0": 0, "y": 1, "n": 0, "Y": 1, "N": 0}).fillna(0).astype(int)
                break
        if lbl is None:
            raise ValueError("Could not find label: expected 'PCOS (Y/N)' or similar")
        df["label"] = lbl.astype(int)

    out = pd.DataFrame(index=df.index)
    out["age"] = get_num("age_(yrs)")
    out["bmi"] = get_num("bmi")
    out["amh"] = get_num("amh(ng/ml)")

    # total_testosterone: true column rarely present; try proxy with LH if missing (document in report)
    tt = get_num("total_testosterone", default=np.nan)
    if tt.isna().all() and has("lh(miu/ml)"):
        tt = get_num("lh(miu/ml)")
    out["total_testosterone"] = tt

    out["hirsutism"] = get_cat_yn("hair_growth(y/n)", yes=1, no=0, default=0)
    out["acne_severity"] = get_cat_yn("pimples(y/n)", yes=1, no=0, default=0)

    if has("cycle(r/i)"):
        cyc = df[inv["cycle(r/i)"]].astype(str).str.strip().str.upper().map({"R": 0, "I": 1})
        out["cycle_regularity"] = cyc.fillna(0).astype(int)
    else:
        out["cycle_regularity"] = 0

    out["avg_cycle_length_days"] = get_num("cycle_length(days)")
    out["fasting_insulin"] = get_num("fasting_insulin", default=np.nan)

    if has("rbs(mg/dl)"):
        out["fasting_glucose"] = get_num("rbs(mg/dl)")
    else:
        out["fasting_glucose"] = np.nan

    out["label"] = df["label"].astype(int).values
    return out

def select_features(df: pd.DataFrame):
    have = [c for c in TARGET_FEATURES if c in df.columns]
    miss = [c for c in TARGET_FEATURES if c not in df.columns]
    if miss:
        print(f"[ehr][warn] missing features requested but not present: {miss}")
    X = df[have].copy()
    return X, have

def compute_metrics(y, p, thr):
    return dict(
        f1=float(f1_score(y, (p >= thr).astype(int))),
        auroc=float(roc_auc_score(y, p)),
        auprc=float(average_precision_score(y, p)),
        thr=float(thr),
    )

def best_thr_f1(y, p):
    ts = np.linspace(0.05, 0.95, 37)
    f1s = [f1_score(y, (p >= t).astype(int)) for t in ts]
    return float(ts[int(np.argmax(f1s))])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-dir", required=True)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    PROC = Path(args.proc_dir)
    PARQ = PROC / "ehr.parquet"
    OUTM = PROC / "models"; OUTM.mkdir(parents=True, exist_ok=True)
    OUTR = PROC / "results_image_only"; OUTR.mkdir(parents=True, exist_ok=True)

    if not PARQ.exists():
        raise FileNotFoundError(f"{PARQ} not found")

    raw = pd.read_parquet(PARQ)
    df = normalize_dataframe(raw)

    # Split features/label
    X_all_raw, used = select_features(df.drop(columns=["label"]))
    y_all = df["label"].astype(int).values

    # Handle NaNs:
    X_all = X_all_raw.astype(float)

    # 1) Drop columns that are entirely NaN
    all_nan_cols = [c for c in X_all.columns if X_all[c].isna().all()]
    if all_nan_cols:
        print(f"[ehr][warn] dropping all-NaN columns: {all_nan_cols}")
        X_all = X_all.drop(columns=all_nan_cols)
        used = [c for c in used if c not in all_nan_cols]

    # 2) Mean-impute remaining NaNs, then final safety fill
    X_all = X_all.fillna(X_all.mean())
    X_all = X_all.fillna(0.0)

    print(f"[ehr] features used: {len(used)} -> {used}")
    print(f"[ehr] samples: {len(X_all)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_all.values, y_all, test_size=args.val_size, random_state=args.seed, stratify=y_all
    )

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, solver="lbfgs")
    clf.fit(Xtr, y_train)

    ptr = clf.predict_proba(Xtr)[:, 1]
    pv = clf.predict_proba(Xv)[:, 1]

    thr_tr = best_thr_f1(y_train, ptr)
    thr_v  = best_thr_f1(y_val,   pv)

    mtr = compute_metrics(y_train, ptr, thr_tr)
    mv  = compute_metrics(y_val,   pv,  thr_v)

    print(f"[ehr] train AUROC={mtr['auroc']:.3f} AUPRC={mtr['auprc']:.3f} F1={mtr['f1']:.3f} thr={mtr['thr']:.2f}")
    print(f"[ehr]  val  AUROC={mv['auroc']:.3f} AUPRC={mv['auprc']:.3f} F1={mv['f1']:.3f} thr={mv['thr']:.2f}")

    import joblib
    joblib.dump(clf, OUTM / "ehr_model.pkl")
    joblib.dump(scaler, OUTM / "ehr_scaler.pkl")
    (OUTR / "ehr_metrics.json").write_text(json.dumps(
        {"train": mtr, "val": mv, "features": used}, indent=2
    ))
    print(f"[ehr] saved -> {OUTM/'ehr_model.pkl'}, {OUTM/'ehr_scaler.pkl'}, {OUTR/'ehr_metrics.json'}")

if __name__ == "__main__":
    main()
