import json, time, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvmodels

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix
)
import matplotlib.pyplot as plt
from tqdm import tqdm
from sam_optimizer import SAM


# ===================== Helpers =====================

def compute_pos_weight(proc_dir: str) -> float:
    """
    Return N_neg / N_pos computed only from the TRAIN split.
    Prints counts to help debug single-class splits.
    """
    PROC = Path(proc_dir)
    m = pd.read_csv(PROC / "manifest.csv")
    l = pd.read_csv(PROC / "labels.csv")
    df = m.merge(l, on="patient_id", how="inner")
    tr = df[df["split"] == "train"]["label"].astype(int)
    n_pos = int((tr == 1).sum())
    n_neg = int((tr == 0).sum())
    print(f"[class-balance] train counts -> pos={n_pos}  neg={n_neg}")
    if n_pos == 0:
        return 1.0
    return max(float(n_neg) / float(n_pos), 1e-3)


def load_labels(proc: Path, labels_csv: Path, ehr_parq: Path) -> pd.DataFrame:
    """
    Priority: labels.csv; fallback to ehr.parquet (auto-detect a label-like column).
    Returns DataFrame with columns: patient_id, label (0/1 int).
    """
    if labels_csv.exists():
        lab = pd.read_csv(labels_csv)
        lab.columns = [c.strip() for c in lab.columns]
        assert "patient_id" in lab.columns and "label" in lab.columns, \
            "labels.csv must have columns: patient_id,label"
        out = lab[["patient_id", "label"]].copy()
        out["label"] = out["label"].astype(int)
        return out

    if ehr_parq.exists():
        ehr = pd.read_parquet(ehr_parq)
        ehr.columns = [c.strip() for c in ehr.columns]
        cand = [c for c in ["pcos","PCOS","diagnosis","Diagnosis","label","Label","target","Target"] if c in ehr.columns]
        if not cand:
            raise RuntimeError(
                "No labels.csv and no label-like column found in ehr.parquet. "
                "Create processed\\labels.csv with columns patient_id,label (0/1)."
            )
        out = ehr[["patient_id", cand[0]]].rename(columns={cand[0]: "label"}).copy()
        out["label"] = (out["label"].astype(str).str.strip().str.lower()
                        .map({"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0,"pcos":1,"non-pcos":0})).astype(int)
        return out

    raise RuntimeError("No labels.csv or ehr.parquet found.")


def metrics_from_scores(y_true, y_prob, thr=0.5):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = auc(fpr, tpr)
    pr_p, pr_r, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(pr_r, pr_p)
    y_hat = (y_prob >= thr).astype(int)
    acc = (y_hat == y_true).mean()
    prec = precision_score(y_true, y_hat, zero_division=0)
    rec = recall_score(y_true, y_hat, zero_division=0)
    f1 = f1_score(y_true, y_hat, zero_division=0)
    return dict(
        auroc=auroc, auprc=auprc, acc=acc, prec=prec, rec=rec, f1=f1,
        fpr=fpr.tolist(), tpr=tpr.tolist(), pr_p=pr_p.tolist(), pr_r=pr_r.tolist()
    )


def tune_threshold(y_true, y_prob, target="f1"):
    best_thr, best = 0.5, -1
    for thr in np.linspace(0.05, 0.95, 181):
        y_hat = (y_prob >= thr).astype(int)
        if target == "recall":
            score = recall_score(y_true, y_hat, zero_division=0)
        elif target == "precision":
            score = precision_score(y_true, y_hat, zero_division=0)
        else:
            score = f1_score(y_true, y_hat, zero_division=0)
        if score > best:
            best, best_thr = score, thr
    return float(best_thr), float(best)


def find_best_threshold_for_accuracy(y, p):
    best_acc, best_thr = 0.0, 0.5
    for thr_ in np.linspace(0.00, 1.00, 201):
        pred = (p >= thr_).astype(int)
        acc = (pred == y).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr_
    return best_thr, best_acc


def plot_curves(y_true, y_prob, out_prefix: str):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob); roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5), dpi=160)
    plt.plot(fpr, tpr, label=f"AUROC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.legend(); plt.tight_layout(); plt.savefig(out_prefix + "_ROC.png"); plt.close()
    # PR
    pr_p, pr_r, _ = precision_recall_curve(y_true, y_prob); pr_auc = auc(pr_r, pr_p)
    plt.figure(figsize=(7, 5), dpi=160)
    plt.plot(pr_r, pr_p, label=f"AUPRC={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR")
    plt.legend(); plt.tight_layout(); plt.savefig(out_prefix + "_PR.png"); plt.close()


# ===================== Dataset =====================

class ImgDS(Dataset):
    def __init__(self, df, proc_root: Path, img_size=224, train=False):
        self.proc_root = proc_root
        self.df = df.reset_index(drop=True)
        self.train = train

        self.t_train = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.t_val = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize(int(img_size * 1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = self.proc_root / r["image_path"]
        img = Image.open(p).convert("L")
        x = self.t_train(img) if self.train else self.t_val(img)
        y = torch.tensor(int(r["label"]), dtype=torch.float32)
        return x, y


# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-dir", default=r"D:\PCOS\pcos-detection\data\processed",
                    help="Folder containing manifest.csv, labels.csv, and images/")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=0)  # Windows-safe default
    ap.add_argument("--sam", action="store_true", help="Enable Sharpness-Aware Minimization")
    ap.add_argument("--sam-rho", type=float, default=0.05)
    ap.add_argument("--sam-adaptive", type=int, default=1, help="1=adaptive, 0=standard")
    args = ap.parse_args()

    PROC = Path(args.proc_dir)
    MANIFEST = PROC / "manifest.csv"
    LABELS_CSV = PROC / "labels.csv"
    EHR_PARQ = PROC / "ehr.parquet"
    OUT = PROC / "results_image_only"; OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR = PROC / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Class balance
    pos_w = compute_pos_weight(args.proc_dir)
    print(f"[class-balance] train positives/negatives -> pos_weight={pos_w:.3f}")

    # Data
    m = pd.read_csv(MANIFEST)
    labs = load_labels(PROC, LABELS_CSV, EHR_PARQ)
    df = m.merge(labs, on="patient_id", how="inner").dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    print(f"[data] samples with labels: {len(df)}")

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val   = df[df["split"] == "val"].reset_index(drop=True)
    df_test  = df[df["split"] == "test"].reset_index(drop=True)

    ds_train = ImgDS(df_train, PROC, img_size=args.img_size, train=True)
    ds_val   = ImgDS(df_val,   PROC, img_size=args.img_size, train=False)
    ds_test  = ImgDS(df_test,  PROC, img_size=args.img_size, train=False)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    # Model
    def build_model():
        m = tvmodels.resnet50(weights=tvmodels.ResNet50_Weights.IMAGENET1K_V2)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(in_f, 1)  # binary logit
        )
        return m

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)

    # Loss (balanced)
    pos_weight_tensor = torch.tensor([pos_w], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Optimizer + One-Cycle
    base_opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer = SAM(model.parameters(), base_opt, rho=args.sam_rho, adaptive=bool(args.sam_adaptive)) if args.sam else base_opt
    steps_per_epoch = max(1, len(dl_train))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer,
        max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch
    )

    best_f1, best_epoch = -1, -1
    best_path = MODEL_DIR / "best_res50.pt"

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        loss_sum, n = 0.0, 0
        all_probs, all_t = [], []
        with torch.set_grad_enabled(train):
            for xb, yb in tqdm(loader, disable=False):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device).view(-1, 1)

                if train:
                    if isinstance(optimizer, SAM):
                        # First forward-backward pass
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        
                        # Second forward-backward pass
                        logits = model(xb)  # at perturbed weights
                        loss = criterion(logits, yb)
                        loss.backward()
                        optimizer.second_step(zero_grad=True)
                    else:
                        # Standard training
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    scheduler.step()
                else:
                    with torch.no_grad():
                        # TTA: hflip + average
                        logits1 = model(xb)
                        xb_flip = torch.flip(xb, dims=[3])
                        logits2 = model(xb_flip)
                        logits = (logits1 + logits2) / 2.0
                        loss = criterion(logits, yb)

                loss_sum += loss.item() * yb.size(0); n += yb.size(0)
                all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
                all_t.append(yb.detach().cpu().numpy())

        probs = np.vstack(all_probs).ravel(); ytrue = np.vstack(all_t).ravel()
        return loss_sum / max(1, n), ytrue, probs

    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, ytr, ptr = run_epoch(dl_train, train=True)
        val_loss, yv, pv = run_epoch(dl_val, train=False)

        thr_f1, val_f1 = tune_threshold(yv, pv, target="f1")
        metrics_v = metrics_from_scores(yv, pv, thr_f1)

        thr_acc, acc_val = find_best_threshold_for_accuracy(yv, pv)
        print(f"[val] best-ACC thr={thr_acc:.2f} acc={acc_val:.3f}")
        try:
            (OUT / "best_acc_threshold.txt").write_text(f"{thr_acc:.4f}\n")
        except Exception as e:
            print("[warn] could not save best_acc_threshold:", e)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_f1": round(metrics_v["f1"], 6),
            "val_auroc": round(metrics_v["auroc"], 6),
            "val_auprc": round(metrics_v["auprc"], 6),
            "thr_f1": round(thr_f1, 4),
            "time_sec": round(time.time() - t0, 2)
        })
        print(f"[{epoch:02d}] loss {train_loss:.4f}/{val_loss:.4f}  "
              f"F1={metrics_v['f1']:.3f}  AUC={metrics_v['auroc']:.3f}  thr(F1)={thr_f1:.2f}")

        # Early stop on F1
        if metrics_v["f1"] > best_f1:
            best_f1, best_epoch = metrics_v["f1"], epoch
            torch.save({"model": model.state_dict(), "thr": thr_f1}, best_path)
        elif epoch - best_epoch >= args.patience:
            print(f"[early-stop] no F1 improvement for {args.patience} epochs.")
            break

    # ===== Evaluate best on TEST =====
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    best_thr = float(ckpt["thr"])

    _, yt, pt = run_epoch(dl_test, train=False)
    test_metrics = metrics_from_scores(yt, pt, best_thr)
    cm = confusion_matrix(yt, (pt >= best_thr).astype(int)).tolist()

    # Save artifacts
    json.dump(
        {"history": history, "test": test_metrics, "threshold": best_thr, "cm": cm},
        open(OUT / "metrics.json", "w"),
        indent=2
    )
    plot_curves(yt, pt, str(OUT / "test"))
    print(f"[done] Best model: {best_path}")
    print(f"[done] Metrics + curves saved under: {OUT}")


if __name__ == "__main__":
    main()
