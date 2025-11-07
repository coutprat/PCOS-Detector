from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from pathlib import Path
import io, json

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models as tvmodels

import joblib

# -------------------- Paths --------------------
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
MODEL_DIR = PROC / "models"
RES_DIR = PROC / "results_image_only"

IMG_CKPT = MODEL_DIR / "best_res50.pt"
CAL_JSON = RES_DIR / "calibration.json"        # {"T": 0.988} or {"temperature": 0.988}
EHR_MODEL = MODEL_DIR / "ehr_model.pkl"
EHR_SCALER = MODEL_DIR / "ehr_scaler.pkl"

# -------------------- App & CORS --------------------
app = FastAPI(title="PCOS Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5173", "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Image model --------------------
_t = 1.0  # temperature (from calibration.json)
def _load_temperature():
    global _t
    if CAL_JSON.exists():
        try:
            j = json.loads(CAL_JSON.read_text())
            _t = float(j.get("T", j.get("temperature", 1.0)))
        except Exception:
            _t = 1.0

def _build_model():
    m = tvmodels.resnet50(weights=tvmodels.ResNet50_Weights.IMAGENET1K_V2)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, 1)
    return m

model_img = _build_model().to(device)

def _remap_state_dict_keys(sd: dict) -> dict:
    """Make checkpoint keys match our model: strip 'module.' and map 'fc.1.*' -> 'fc.*'."""
    new_sd = {}
    for k, v in sd.items():
        # strip DistributedDataParallel prefix
        if k.startswith("module."):
            k = k[len("module."):]
        # map sequential head keys to plain fc
        if k.startswith("fc.1."):
            k = "fc." + k[len("fc.1."):]
        new_sd[k] = v
    return new_sd

def _load_image_ckpt() -> bool:
    if not IMG_CKPT.exists():
        return False
    ck = torch.load(IMG_CKPT, map_location=device)
    # accept raw state_dict or {"model": state_dict}
    if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
        sd = ck["model"]
    else:
        sd = ck if isinstance(ck, dict) else {}
    if not sd:
        return False
    sd = _remap_state_dict_keys(sd)
    # be tolerant to any leftover extras
    model_img.load_state_dict(sd, strict=False)
    model_img.eval()
    return True

_loaded_img = _load_image_ckpt()
_load_temperature()

img_tf = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def _img_prob(img: Image.Image) -> float:
    x = img_tf(img.convert("L")).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model_img(x) / (_t if _t > 0 else 1.0)
        p = torch.sigmoid(logit).item()
    return float(p)

# -------------------- EHR model --------------------
ehr_clf = None
ehr_scaler = None
ehr_features = [
    "age","bmi","amh","total_testosterone","hirsutism","acne_severity",
    "cycle_regularity","avg_cycle_length_days","fasting_glucose"  # (fasting_insulin dropped as all-NaN)
]

def _try_load_ehr() -> bool:
    global ehr_clf, ehr_scaler
    if EHR_MODEL.exists() and EHR_SCALER.exists():
        ehr_clf = joblib.load(EHR_MODEL)
        ehr_scaler = joblib.load(EHR_SCALER)
        return True
    return False

_loaded_ehr = _try_load_ehr()

def _ehr_from_payload(ehr: Dict[str, Any]) -> float:
    dem  = ehr.get("demographics", {})
    endo = ehr.get("endocrine", {})
    mens = ehr.get("menstrual", {})
    metab= ehr.get("metabolic", {})

    row = {
        "age": dem.get("age"),
        "bmi": dem.get("bmi"),
        "amh": endo.get("amh"),
        "total_testosterone": endo.get("total_testosterone"),
        "hirsutism": endo.get("hirsutism"),
        "acne_severity": endo.get("acne_severity"),
        "cycle_regularity": 1 if str(mens.get("cycle_regularity","")).lower()=="irregular" else 0,
        "avg_cycle_length_days": mens.get("avg_cycle_length_days"),
        "fasting_glucose": metab.get("fasting_glucose"),
    }
    X = pd.DataFrame([row], columns=ehr_features).astype(float)
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
    X = X.fillna(X.mean()).fillna(0.0)

    model_cols = getattr(ehr_clf, "feature_names_in_", None)
    if model_cols is not None:
        Xa = pd.DataFrame({c: X[c] if c in X.columns else 0.0 for c in model_cols})
        X_use = Xa.values
    else:
        X_use = X.values

    Xs = ehr_scaler.transform(X_use) if ehr_scaler is not None else X_use
    p = float(ehr_clf.predict_proba(Xs)[0,1]) if ehr_clf is not None else 0.5
    return p

# -------------------- Schemas --------------------
class HealthResp(BaseModel):
    status: str
    device: str
    ckpt_exists: bool
    ehr_exists: bool
    temperature: float

# -------------------- Routes --------------------
@app.get("/health", response_model=HealthResp)
def health():
    return HealthResp(
        status="ok",
        device=str(device),
        ckpt_exists=IMG_CKPT.exists(),
        ehr_exists=EHR_MODEL.exists() and EHR_SCALER.exists(),
        temperature=float(_t),
    )

@app.post("/reload")
def reload_models():
    ok_img = _load_image_ckpt()
    _load_temperature()
    ok_ehr = _try_load_ehr()
    return {"reloaded": bool(ok_img or ok_ehr)}

@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    img = Image.open(io.BytesIO(await file.read())).convert("L")
    p = _img_prob(img)
    label = "PCOS Positive" if p >= threshold else "PCOS Negative"
    return {
        "probability": p,
        "label": label,
        "threshold": float(threshold),
        "checkpoint_loaded": _loaded_img,
        "temperature": _t,
    }

@app.post("/predict_ehr")
async def predict_ehr(ehr: Dict[str, Any]):
    if ehr_clf is None:
        return {"error": "ehr model not loaded"}
    p = _ehr_from_payload(ehr)
    label = "PCOS Positive" if p >= 0.5 else "PCOS Negative"
    return {"probability": p, "label": label, "threshold": 0.5}

@app.post("/predict_hybrid")
async def predict_hybrid(
    file: UploadFile = File(...),
    ehr_json: str = Form(...),
    threshold: float = 0.5,
    w_img: float = 0.6
):
    if ehr_clf is None:
        return {"error": "ehr model not loaded"}

    img = Image.open(io.BytesIO(await file.read())).convert("L")
    p_img = _img_prob(img)

    try:
        ehr = json.loads(ehr_json)
    except Exception:
        return {"error": "invalid ehr payload"}
    p_ehr = _ehr_from_payload(ehr)

    w = float(w_img)
    p = w * p_img + (1.0 - w) * p_ehr
    label = "PCOS Positive" if p >= threshold else "PCOS Negative"
    return {
        "probability": float(p),
        "label": label,
        "threshold": float(threshold),
        "image_prob": float(p_img),
        "ehr_prob": float(p_ehr),
        "w_img": w,
    }
