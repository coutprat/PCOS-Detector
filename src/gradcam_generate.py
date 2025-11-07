import argparse, os, json, random, platform
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models

# ---------- args ----------
ap = argparse.ArgumentParser()
ap.add_argument("--proc-dir", default=r"D:\PCOS\pcos-detection\data\processed")
ap.add_argument("--n", type=int, default=24, help="number of test images to visualize")
ap.add_argument("--img-size", type=int, default=224)
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()
if platform.system() == "Windows": args.workers = 0

PROC = Path(args.proc_dir)
MANIFEST = PROC/"manifest.csv"
LABELS   = PROC/"labels.csv"
CKPT     = PROC/"models"/"best_effb0.pt"
OUTDIR   = PROC/"results_image_only"/"gradcam"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- data ----------
class ValDS(Dataset):
    def __init__(self, df, proc_root, img_size):
        self.df = df.reset_index(drop=True); self.proc_root = proc_root
        self.tf = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r = self.df.iloc[i]
        p = self.proc_root / r["image_path"]
        x = self.tf(Image.open(p).convert("L"))
        y = int(r["label"])
        return x, y, str(p)

m = pd.read_csv(MANIFEST)
lab = pd.read_csv(LABELS)
df = m.merge(lab, on="patient_id", how="inner")
df_test = df[df["split"]=="test"].reset_index(drop=True)
if len(df_test)==0: raise SystemExit("No test rows after merge.")

# sample a subset
if len(df_test) > args.n: df_test = df_test.sample(args.n, random_state=42).reset_index(drop=True)

ds = ValDS(df_test, PROC, args.img_size)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

# ---------- model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
model = models.efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
ck = torch.load(CKPT, map_location=device)
model.load_state_dict(ck["model"])
model.eval().to(device)
best_thr = float(ck.get("thr", 0.5))

# find last conv layer inside features
target_layer = None
for m_ in reversed(model.features):
    if isinstance(m_, nn.Sequential):
        for mm in reversed(m_):
            if isinstance(mm, nn.Conv2d):
                target_layer = mm; break
    if target_layer: break
if target_layer is None:
    # fallback: any conv
    for mm in model.modules():
        if isinstance(mm, nn.Conv2d): target_layer = mm
assert target_layer is not None, "No Conv2d layer found."

feats = []
grads = []
def f_hook(_, __, output): feats.append(output.detach())
def b_hook(_, grad_in, grad_out): grads.append(grad_out[0].detach())
target_layer.register_forward_hook(f_hook)
# replace the next line with a safe registration that works on older/newer PyTorch
try:
    target_layer.register_full_backward_hook(b_hook)
except AttributeError:
    target_layer.register_backward_hook(b_hook)

# ---------- make CAM ----------
def make_cam(feat, grad):
    # feat: (1,C,H,W), grad: (1,C,H,W)
    w = grad.mean(dim=(2,3), keepdim=True)   # GAP over H,W
    cam = (w * feat).sum(dim=1, keepdim=True)  # (1,1,H,W)
    cam = cam.relu()
    cam = cam / (cam.max() + 1e-6)
    return cam[0,0].cpu().numpy()

import matplotlib.pyplot as plt

with torch.no_grad():
    pass  # ensure no grads persist

for xb, yb, path_str in dl:
    feats.clear(); grads.clear()
    xb = xb.to(device).requires_grad_(True)

    logits = model(xb)  # shape: [1,1]
    prob = torch.sigmoid(logits).item()
    model.zero_grad(set_to_none=True)
    logits.backward(torch.ones_like(logits))

    # if path_str is a list (batch), take the first element
    pstr = path_str if isinstance(path_str, str) else path_str[0]
    base = Path(pstr).stem

    feat = feats[-1]; grad = grads[-1]
    cam = make_cam(feat, grad)   # (H,W) in feature space

    # upscale cam to image size
    cam_img = Image.fromarray((cam*255).astype(np.uint8)).resize((args.img_size,args.img_size), Image.BILINEAR)
    cam_arr = np.array(cam_img)/255.0

    # get original (denormalized) grayscale
    x = xb[0].detach().cpu()
    # x is 3xHxW normalized; take one channel and denorm
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    x_den = (x*std+mean)[0].clamp(0,1).numpy()

    # overlay
    plt.figure(figsize=(4,4), dpi=200)
    plt.imshow(x_den, cmap="gray", interpolation="nearest")
    plt.imshow(cam_arr, cmap="jet", alpha=0.35, interpolation="bilinear")
    pred = 1 if prob >= best_thr else 0
    plt.title(f"{base}  prob={prob:.2f}  thr={best_thr:.2f}  pred={pred}")
    plt.axis("off")
    outpath = OUTDIR / f"cam_{base}.png"
    plt.tight_layout(); plt.savefig(outpath); plt.close()

print(f"[gradcam] saved overlays to: {OUTDIR}")
