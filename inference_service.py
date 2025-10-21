#!/usr/bin/env python3
"""
inference_service.py

Load TorchScript models + sklearn meta-learner + thresholds and run inference on a single image
(or on a folder of images). Designed to be used by FastAPI later.

Usage examples (CLI):
 python inference_service.py --image "C:\Varun\skin-detect-mvp\data\test\bcc\ISIC_0024345.jpg"
 python inference_service.py --image_dir "C:\Varun\skin-detect-mvp\data\test" --limit 10

Output:
 Prints JSON for single image or a small summary CSV for folder mode.
"""
import argparse, json, os, time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import joblib

BASE = Path(r"C:\Varun\skin-detect-mvp")
EXPORT_DIR = BASE / "export"
METADATA_P = EXPORT_DIR / "metadata.json"
META_JOBLIB = EXPORT_DIR / "meta_learner.joblib"

if not METADATA_P.exists():
    raise RuntimeError(f"Missing metadata.json at {METADATA_P}. Run export first.")

with open(METADATA_P, "r", encoding="utf8") as f:
    metadata = json.load(f)

# map of kind -> torchscript path & image_size
MODEL_INFO = {}
for k in ("b0","b3","resnet50"):
    entry = metadata.get(k)
    if entry:
        MODEL_INFO[k] = {
            "script": entry.get("torchscript"),
            "image_size": entry.get("image_size", 224)
        }

CLASS_NAMES = metadata.get("class_names", None)
THRESHOLDS = metadata.get("thresholds", None) or {}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load TorchScript modules lazily (to free memory between loads if needed)
def load_script_module(path):
    path = str(path)
    if not Path(path).exists():
        raise RuntimeError(f"TorchScript model missing: {path}")
    m = torch.jit.load(path, map_location=DEVICE)
    m.eval()
    return m

def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def run_models_sequential(image_path, tta_views=1, use_gpu=True):
    """
    For given image_path, run each model sequentially to avoid holding all models in GPU at once.
    Returns dict {model_kind: probs_array}, where probs_array is 1D numpy array (C,)
    """
    image = Image.open(image_path).convert("RGB")
    model_probs = {}
    for kind, info in MODEL_INFO.items():
        script_path = info["script"]
        img_sz = int(info.get("image_size", 224))
        tfm = get_transform(img_sz)
        x = tfm(image).unsqueeze(0)  # 1,C,H,W
        # load model
        mod = load_script_module(script_path)
        mod = mod.to(DEVICE)
        with torch.no_grad():
            probs_views = []
            # original
            logits = mod(x.to(DEVICE))
            probs_views.append(torch.softmax(logits, dim=1).cpu().numpy().squeeze())
            if tta_views >= 2:
                flipped = torch.flip(x, dims=[3])
                logits = mod(flipped.to(DEVICE))
                probs_views.append(torch.softmax(logits, dim=1).cpu().numpy().squeeze())
            if tta_views >= 3:
                vfl = torch.flip(x, dims=[2])
                logits = mod(vfl.to(DEVICE))
                probs_views.append(torch.softmax(logits, dim=1).cpu().numpy().squeeze())
            avg = np.mean(np.vstack(probs_views), axis=0)
        # free model explicitly
        del mod
        torch.cuda.empty_cache()
        model_probs[kind] = avg
    return model_probs

def stack_and_meta_predict(model_probs_dict):
    """
    model_probs_dict: dict of kind->(C,) arrays in same class order as class_names
    We will concatenate in sorted order of MODEL_INFO keys (deterministic).
    """
    # Build stacked vector
    kinds = sorted(list(model_probs_dict.keys()))
    arrs = [model_probs_dict[k] for k in kinds]
    stacked = np.concatenate(arrs, axis=0).reshape(1, -1)  # 1 x (M*C)
    # load meta-learner
    clf = joblib.load(META_JOBLIB)
    pred = clf.predict(stacked)[0]
    proba = clf.predict_proba(stacked)[0]  # order: clf.classes_
    # map proba to canonical class order if needed
    classes_meta = list(clf.classes_)
    # prepare dict of class->prob (meta prediction)
    meta_probs = {c: float(p) for c,p in zip(classes_meta, proba)}
    # choose final label using thresholds if provided
    chosen = None
    chosen_prob = 0.0
    if THRESHOLDS:
        exceeded = [(meta_probs[c], c) for c in classes_meta if meta_probs.get(c,0.0) >= THRESHOLDS.get(c, 0.5)]
        if exceeded:
            chosen = max(exceeded, key=lambda x: x[0])[1]
            chosen_prob = float(max(exceeded)[0])
    if chosen is None:
        idx = int(np.argmax(proba))
        chosen = classes_meta[idx]
        chosen_prob = float(proba[idx])
    return {
        "pred_label": chosen,
        "pred_prob": chosen_prob,
        "meta_probs": meta_probs,
        "meta_classes": classes_meta
    }

def infer_single(image_path, tta=1):
    t0 = time.time()
    model_probs = run_models_sequential(image_path, tta_views=tta)
    stacked_result = stack_and_meta_predict(model_probs)
    elapsed = time.time() - t0
    # also include raw per-model probs for debugging
    result = {
        "image": str(image_path),
        "device": DEVICE,
        "elapsed_seconds": elapsed,
        "per_model_probs": {k: model_probs[k].tolist() for k in sorted(model_probs.keys())},
        "stack_result": stacked_result
    }
    return result

# CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, help="Path to single image")
    p.add_argument("--image_dir", type=str, help="Folder to process (writes csv)")
    p.add_argument("--tta", type=int, default=1, choices=[1,2,3], help="TTA views")
    p.add_argument("--limit", type=int, default=-1, help="limit images when using image_dir")
    args = p.parse_args()

    if args.image:
        out = infer_single(Path(args.image), tta=args.tta)
        print(json.dumps(out, indent=2))
    elif args.image_dir:
        import csv
        files = sorted(list(Path(args.image_dir).rglob("*.jpg")) + list(Path(args.image_dir).rglob("*.png")))
        if args.limit > 0:
            files = files[:args.limit]
        out_rows = []
        for f in files:
            res = infer_single(f, tta=args.tta)
            pred = res["stack_result"]["pred_label"]
            prob = res["stack_result"]["pred_prob"]
            out_rows.append((str(f), pred, prob))
            print(f"{f.name} -> {pred} ({prob:.3f})  time: {res['elapsed_seconds']:.2f}s")
        # write CSV
        csv_p = BASE / "predictions_service_batch.csv"
        with open(csv_p, "w", newline="", encoding="utf8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["filename","pred","pred_prob"])
            writer.writerows(out_rows)
        print("Saved batch CSV ->", csv_p)
    else:
        print("Provide --image or --image_dir. Example:\n python inference_service.py --image \"C:\\...jpg\" --tta 2")
