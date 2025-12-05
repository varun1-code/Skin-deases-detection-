# app.py (updated)
import os, io, json, time, traceback, base64, smtplib
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import joblib
import email.message
import urllib.parse

BASE = Path(__file__).resolve().parent
EXPORT = BASE / "export"
METADATA_P = EXPORT / "metadata.json"
CASES_FILE = BASE / "cases.jsonl"

if not METADATA_P.exists():
    raise RuntimeError(f"metadata.json not found at {METADATA_P}. Run export_models_torchscript.py first.")

with open(METADATA_P, "r", encoding="utf8") as f:
    METADATA = json.load(f)

# Simple config from env
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMTP_HOST = os.environ.get("SB_SMTP_HOST")
SMTP_PORT = int(os.environ.get("SB_SMTP_PORT", 587)) if os.environ.get("SB_SMTP_HOST") else None
SMTP_USER = os.environ.get("SB_SMTP_USER")
SMTP_PASS = os.environ.get("SB_SMTP_PASS")
FROM_EMAIL = os.environ.get("SB_FROM_EMAIL", "noreply@example.com")
CLINIC_EMAIL = os.environ.get("SB_CLINIC_EMAIL", "appointments@exampleclinic.org")
ADMIN_TOKEN = os.environ.get("SB_ADMIN_TOKEN", "localadmintoken")

# Load meta learner
META_JOBLIB = EXPORT / "meta_learner.joblib"
if not META_JOBLIB.exists():
    raise RuntimeError("meta_learner.joblib missing in export folder.")
meta_clf = joblib.load(str(META_JOBLIB))

# Models cache
TS_CACHE: Dict[str, torch.jit.ScriptModule] = {}
TIMM_CACHE: Dict[str, torch.nn.Module] = {}

# helpers
def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def save_case(case: Dict[str,Any]):
    with open(CASES_FILE, "a", encoding="utf8") as f:
        f.write(json.dumps(case) + "\n")

def load_torchscript_model(key:str):
    """Load TorchScript model path from metadata key name (e.g. 'b0','b3','resnet50')."""
    info = METADATA.get(key)
    if not info:
        raise RuntimeError(f"No model entry for {key} in metadata.")
    path = Path(info["torchscript"])
    if not path.exists():
        raise RuntimeError(f"TorchScript model missing: {path}")
    if key in TS_CACHE:
        return TS_CACHE[key]
    m = torch.jit.load(str(path), map_location=DEVICE)
    m.eval()
    TS_CACHE[key] = m
    return m

def get_default_transform(image_size:int):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def ensemble_predict_from_pil(img_pil:Image.Image, tta:int=2):
    per_model = {}
    for key, info in METADATA.items():
        if "torchscript" not in info:
            continue
        img_sz = int(info.get("image_size", 224))
        model = load_torchscript_model(key).to(DEVICE)
        tfm = get_default_transform(img_sz)
        x = tfm(img_pil).unsqueeze(0).to(DEVICE)
        probs_accum = []
        with torch.no_grad():
            out = model(x)
            probs_accum.append(F.softmax(out, dim=1).cpu().numpy().squeeze())
            if tta >= 2:
                out = model(torch.flip(x, dims=[3]))
                probs_accum.append(F.softmax(out, dim=1).cpu().numpy().squeeze())
            if tta >= 3:
                out = model(torch.flip(x, dims=[2]))
                probs_accum.append(F.softmax(out, dim=1).cpu().numpy().squeeze())
        avg = np.mean(np.vstack(probs_accum), axis=0)
        per_model[key] = avg.tolist()
        torch.cuda.empty_cache()
    # meta stack order: use sorted keys for consistency
    keys_sorted = sorted(per_model.keys())
    stacked = np.concatenate([np.array(per_model[k]) for k in keys_sorted], axis=0).reshape(1,-1)
    pred_label = meta_clf.predict(stacked)[0]
    meta_probs = {c: float(p) for c,p in zip(list(meta_clf.classes_), meta_clf.predict_proba(stacked)[0])}
    return {"per_model_probs": per_model, "pred_label": pred_label, "meta_probs": meta_probs, "meta_classes": list(meta_clf.classes_)}

# --- Grad-CAM utilities ---
def try_load_timm_model_from_metadata(key: str):
    """Best-effort: load a timm model using metadata 'model_name' or fallback guesses; returns model or None."""
    info = METADATA.get(key, {})
    model_name = info.get("model_name") or info.get("timm_name")
    if not model_name:
        return None
    try:
        import timm
        if key in TIMM_CACHE:
            return TIMM_CACHE[key]
        m = timm.create_model(model_name, pretrained=False, num_classes=len(meta_clf.classes_))
        # If there is a checkpoint path in metadata, try loading weights
        ckpt = info.get("checkpoint")  # not guaranteed
        if ckpt and Path(ckpt).exists():
            ck = torch.load(ckpt, map_location="cpu")
            # try to load 'model_state' or entire state_dict
            sd = ck.get("model_state", ck)
            m.load_state_dict(sd, strict=False)
        m.eval()
        m.to(DEVICE)
        TIMM_CACHE[key] = m
        return m
    except Exception:
        return None

def compute_gradcam_pil(img_pil: Image.Image, model_key: str, target_class_idx: Optional[int]=None):
    """Return overlay PNG bytes (RGBA) as base64-ready bytes. Two methods:
       - Prefer timm model with layer access (GradCAM)
       - Fallback: input gradients on TorchScript model to produce saliency map (coarse)
    """
    # try timm model
    timm_model = try_load_timm_model_from_metadata(model_key)
    img_sz = int(METADATA.get(model_key, {}).get("image_size", 224))
    tfm = get_default_transform(img_sz)
    x = tfm(img_pil).unsqueeze(0).to(DEVICE).requires_grad_(True)

    if timm_model is not None:
        # simple grad-cam: grab last conv by heuristic
        last_conv = None
        for name, module in reversed(list(timm_model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                last_conv = (name, module)
                break
        if last_conv is None:
            # fall back to gradient saliency
            timm_model.zero_grad()
            logits = timm_model(x)
            cls = int(target_class_idx) if target_class_idx is not None else int(logits.argmax(dim=1).item())
            loss = logits[0, cls]
            loss.backward(retain_graph=False)
            sal = x.grad.abs().sum(dim=1).squeeze().cpu().numpy()
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
            sal = np.clip(sal, 0, 1)
            heat = (sal * 255).astype("uint8")
        else:
            # register forward hook to capture activations
            activations = {}
            def forward_hook(module, inp, out):
                activations['feat'] = out.detach()
            handle = dict(timm_model.named_modules())[last_conv[0]].register_forward_hook(forward_hook)
            timm_model.zero_grad()
            logits = timm_model(x)
            cls = int(target_class_idx) if target_class_idx is not None else int(logits.argmax(dim=1).item())
            loss = logits[0, cls]
            loss.backward()
            feat = activations['feat'][0]  # C x H x W
            weights = x.grad.mean(dim=(2,3)).squeeze().cpu().numpy() if x.grad is not None else np.ones(feat.shape[0])
            cam = np.zeros(feat.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * feat[i].cpu().numpy()
            cam = np.maximum(cam, 0)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
            heat = (np.uint8(255 * cam))
            handle.remove()
        # upsample heat to image size
        heat_img = Image.fromarray(heat).resize(img_pil.size, resample=Image.BILINEAR).convert("L")
    else:
        # fallback using torchscript saliency
        # compute gradients of input w.r.t. top logit
        ts_model = load_torchscript_model(next(iter(METADATA.keys())))  # pick any TS model
        ts_model = ts_model.to(DEVICE)
        tfm2 = get_default_transform(img_sz)
        x2 = tfm2(img_pil).unsqueeze(0).to(DEVICE).requires_grad_(True)
        ts_model.zero_grad()
        out = ts_model(x2)
        cls = int(target_class_idx) if target_class_idx is not None else int(out.argmax(dim=1).item())
        loss = out[0, cls]
        loss.backward()
        sal = x2.grad.abs().sum(dim=1).squeeze().cpu().numpy()
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
        heat_img = Image.fromarray(np.uint8(sal*255)).resize(img_pil.size, resample=Image.BILINEAR).convert("L")

    # overlay heatmap on original
    orig = img_pil.convert("RGBA")
    cmap = np.array(heat_img).astype("float32")
    # convert to RGBA heat (red)
    heat_rgba = np.zeros((cmap.shape[0], cmap.shape[1], 4), dtype=np.uint8)
    heat_rgba[...,0] = np.clip(cmap*1.0,0,255).astype(np.uint8)
    heat_rgba[...,1] = 0
    heat_rgba[...,2] = 0
    heat_rgba[...,3] = np.clip((cmap*0.6),0,255).astype(np.uint8)  # alpha
    heat_pil = Image.fromarray(heat_rgba, mode="RGBA")
    overlay = Image.alpha_composite(orig, heat_pil)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return buf.getvalue()

# --- FastAPI app ---
app = FastAPI(title="SkinDetect - integrated API")

class PredictResp(BaseModel):
    per_model_probs: dict
    pred_label: str
    meta_probs: dict
    meta_classes: list

@app.post("/predict")
async def predict(file: UploadFile = File(...), tta: int = Form(2)):
    try:
        contents = await file.read()
        pil = pil_from_bytes(contents)
        start = time.time()
        res = ensemble_predict_from_pil(pil, tta=tta)
        elapsed = time.time() - start
        # save a case summary (no user info here)
        case = {
            "timestamp": time.time(),
            "filename": file.filename,
            "pred_label": res["pred_label"],
            "meta_probs_topk": sorted(res["meta_probs"].items(), key=lambda x:-x[1])[:3],
            "tta": tta
        }
        save_case(case)
        return JSONResponse({"elapsed_seconds": elapsed, "result": res})
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(e) + "\n" + tb)

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...), model_key: Optional[str] = Form(None), target_class: Optional[int] = Form(None)):
    """Return base64 PNG overlay as JSON { 'overlay_b64': 'data:image/png;base64,...' }"""
    try:
        contents = await file.read()
        pil = pil_from_bytes(contents)
        # choose model_key fallback to first model in metadata
        if model_key is None:
            model_key = list(METADATA.keys())[0]
        png_bytes = compute_gradcam_pil(pil, model_key, target_class)
        b64 = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        return {"overlay_b64": b64}
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(e) + "\n" + tb)

class BookingReq(BaseModel):
    user: str
    patient_age: Optional[str] = None
    patient_sex: Optional[str] = None
    address: Optional[str] = None
    pred_label: Optional[str] = None
    notes: Optional[str] = None
    contact_email: Optional[str] = None

@app.post("/book")
async def book(req: BookingReq):
    """Send booking request via SMTP (if configured) or save to local file."""
    try:
        # Build message body
        body = f"""Booking request from CareBot:
User: {req.user}
Age: {req.patient_age}
Sex: {req.patient_sex}
Address: {req.address}
Predicted label: {req.pred_label}
Notes: {req.notes}
Contact email: {req.contact_email}
Timestamp: {datetime.datetime.utcnow().isoformat() if 'datetime' in globals() else time.time()}
"""
        saved = {"user": req.user, "timestamp": time.time(), "payload": req.dict()}
        # Save locally
        with open(BASE / "bookings.jsonl", "a", encoding="utf8") as f:
            f.write(json.dumps(saved) + "\n")
        # If SMTP configured, send email
        if SMTP_HOST and SMTP_USER and SMTP_PASS:
            msg = email.message.EmailMessage()
            msg["From"] = FROM_EMAIL
            msg["To"] = CLINIC_EMAIL
            msg["Subject"] = f"CareBot booking: {req.user} - {req.pred_label or 'lesion review'}"
            msg.set_content(body)
            s = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
            s.quit()
            return {"status":"sent","method":"smtp"}
        else:
            return {"status":"saved","method":"local"}
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(e) + "\n" + tb)

@app.get("/admin/cases")
async def admin_cases(token: Optional[str] = None):
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized - invalid admin token")
    # read cases
    out = []
    if CASES_FILE.exists():
        with open(CASES_FILE, "r", encoding="utf8") as f:
            for ln in f:
                try:
                    out.append(json.loads(ln.strip()))
                except Exception:
                    continue
    return {"count": len(out), "cases": out}
