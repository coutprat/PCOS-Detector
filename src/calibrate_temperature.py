import argparse, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, f1_score,
                             precision_score, recall_score, brier_score_loss)
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--proc-dir", default=r"D:\PCOS\pcos-detection\data\processed")
ap.add_argument("--img-size", type=int, default=224)
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()
if platform.system() == "Windows": args.workers = 0

PROC = Path(args.proc_dir)
MANIFEST = PROC/"manifest.csv"
LABELS   = PROC/"labels.csv"
CKPT     = PROC/"models"/"best_effb0.pt"
OUTDIR   = PROC/"results_image_only"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- data ----------
class ValDS(Dataset):
    def __init__(self, df, proc_root, img_size):
        self.df = df.reset_index(drop=True); self.proc_root = proc_root
        self.tf = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize(img_size), T.CenterCrop(img_size),
            T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r = self.df.iloc[i]
        p = self.proc_root / r["image_path"]
        x = self.tf(Image.open(p).convert("L"))
        y = int(r["label"])
        return x, y

def collect_logits(df):
    ds = ValDS(df, PROC, args.img_size)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)
    logits_all, y_all = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            lg = model(xb).squeeze(1).cpu()
            logits_all.append(lg)
            y_all.append(yb)
    return torch.cat(logits_all), torch.cat(y_all)

def metrics(y_true, probs, thr):
    y_true = y_true.numpy()
    p = probs.numpy()
    fpr,tpr,_=roc_curve(y_true,p); auroc=auc(fpr,tpr)
    pr_p,pr_r,_=precision_recall_curve(y_true,p); auprc=auc(pr_r,pr_p)
    yhat=(p>=thr).astype(int)
    return {
        "auroc":float(auroc),"auprc":float(auprc),
        "acc":float((yhat==y_true).mean()),
        "prec":float(precision_score(y_true,yhat,zero_division=0)),
        "rec":float(recall_score(y_true,yhat,zero_division=0)),
        "f1":float(f1_score(y_true,yhat,zero_division=0)),
        "brier":float(brier_score_loss(y_true,p))
    }

def reliability_diagram(y_true, probs, path, bins=10):
    y_true = y_true.numpy(); p = probs.numpy()
    edges = np.linspace(0,1,bins+1); mids=(edges[:-1]+edges[1:])/2
    accs=[]; confs=[]
    for a,b in zip(edges[:-1], edges[1:]):
        m = (p>=a)&(p<b)
        if m.sum()==0: accs.append(np.nan); confs.append((a+b)/2)
        else:
            accs.append(y_true[m].mean()); confs.append((a+b)/2)
    plt.figure(figsize=(5,5), dpi=160)
    plt.plot([0,1],[0,1],'--')
    plt.plot(confs, accs, marker='o')
    plt.xlabel("Confidence"); plt.ylabel("Empirical Accuracy"); plt.title("Reliability Diagram")
    plt.tight_layout(); plt.savefig(path); plt.close()

# ---------- model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
model = models.efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
ck = torch.load(CKPT, map_location=device)
model.load_state_dict(ck["model"])
model.eval().to(device)
best_thr = float(ck.get("thr",0.5))

# ---------- dataframes ----------
m = pd.read_csv(MANIFEST)
l = pd.read_csv(LABELS)
df = m.merge(l, on="patient_id", how="inner")
df_val  = df[df["split"]=="val"].reset_index(drop=True)
df_test = df[df["split"]=="test"].reset_index(drop=True)
if len(df_val)==0 or len(df_test)==0:
    raise SystemExit("Need non-empty val and test splits.")

# ---------- collect logits ----------
with torch.no_grad():
    lg_val, yv = collect_logits(df_val)
    lg_test, yt = collect_logits(df_test)

# ---------- fit temperature on val (optimize NLL) ----------
T = torch.ones(1, requires_grad=True, device=device)
crit = nn.BCEWithLogitsLoss()
opt  = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

yv_t = torch.tensor(yv.numpy(), dtype=torch.float32, device=device).view(-1,1)
lg_val_d = lg_val.to(device).view(-1,1)

def closure():
    opt.zero_grad(set_to_none=True)
    loss = crit(lg_val_d / T.clamp(min=1e-3), yv_t)
    loss.backward()
    return loss

opt.step(closure)
T_star = float(T.data.clamp(min=1e-3).cpu())

# ---------- evaluate before/after ----------
pv_raw = torch.sigmoid(lg_test).cpu().view(-1)
pv_cal = torch.sigmoid(lg_test / T_star).cpu().view(-1)

def tune_thr(y_true, p):
    best,thr=-1,0.5
    for t in np.linspace(0.05,0.95,181):
        f1 = f1_score(y_true.numpy(), (p.numpy()>=t).astype(int), zero_division=0)
        if f1>best: best,thr=f1,t
    return float(thr)

thr_raw = tune_thr(yt, pv_raw)
thr_cal = tune_thr(yt, pv_cal)

m_raw = metrics(yt, pv_raw, thr_raw)
m_cal = metrics(yt, pv_cal, thr_cal)

json.dump({"temperature":T_star, "raw":m_raw, "calibrated":m_cal,
           "thr_raw":thr_raw, "thr_cal":thr_cal},
          open(OUTDIR/"calibration.json","w"), indent=2)

reliability_diagram(yt, pv_raw, OUTDIR/"reliability_raw.png")
reliability_diagram(yt, pv_cal, OUTDIR/"reliability_calibrated.png")
print(f"[calibration] T*={T_star:.3f}  raw F1={m_raw['f1']:.3f}  cal F1={m_cal['f1']:.3f}")
print(f"Saved: {OUTDIR/'calibration.json'}, reliability_raw.png, reliability_calibrated.png")
