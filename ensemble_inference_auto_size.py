#!/usr/bin/env python3
"""
ensemble_inference_auto_size.py

Ensemble inference with per-model image-size auto-detection and sequential model loading.

Usage example:
 python ensemble_inference_auto_size.py \
   --ckpts "C:\Varun\skin-detect-mvp\outputs\finetune_best_epoch2_auc0.9565.pt" "C:\Varun\skin-detect-mvp\outputs\best_efficientnet_b3_epoch10_auc0.9713.pt" "C:\Varun\skin-detect-mvp\outputs\best_resnet50_epoch11_auc0.9665.pt" \
   --input "C:\Varun\skin-detect-mvp\data\test" \
   --out_csv "C:\Varun\skin-detect-mvp\predictions_ensemble.csv" \
   --device cuda --tta_views 3

You may optionally pass --image_sizes 224 300 224 to override auto-detection.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import csv
import tempfile
import os
from sklearn.metrics import classification_report, confusion_matrix

# try timm
try:
    import timm
    _HAS_TIMM = True
except Exception:
    timm = None
    _HAS_TIMM = False

# -------------------------
# Utilities
# -------------------------
def infer_image_size_from_ckpt_path(ckpt_path):
    """Heuristic to pick an image size from checkpoint filename"""
    name = str(ckpt_path).lower()
    if 'efficientnet_b3' in name or '_b3' in name or 'eb3' in name:
        return 300
    if 'efficientnet_b0' in name or '_b0' in name or 'eb0' in name:
        return 224
    if 'resnet' in name or 'resnet50' in name:
        return 224
    # fallback
    return 224

def try_load_state_dict_safely(model, state_dict):
    """
    Try to load state_dict exactly. Return (ok:bool, message:str)
    If exact load fails, returns False.
    """
    try:
        model.load_state_dict(state_dict)
        return True, "loaded_strict"
    except Exception as e:
        return False, str(e)

def build_model_from_ckpt(ckpt_path, device):
    """
    Robustly build a model that matches the checkpoint architecture.
    Tries a list of candidate timm models and torchvision.resnet50 until
    one successfully loads the checkpoint's state_dict.

    Returns (model, class_to_idx, model_name_used)
    """
    ck = torch.load(ckpt_path, map_location='cpu')
    class_to_idx = ck.get('class_to_idx', None)
    if class_to_idx is None:
        raise RuntimeError(f"Checkpoint {ckpt_path} missing 'class_to_idx' mapping.")
    state = ck.get('model_state', None)
    if state is None:
        # maybe checkpoint is just a state_dict
        state = ck
    num_classes = len(class_to_idx)

    tried = []

    # Candidate timm names to try (order matters: common ones first)
    timm_candidates = []
    if _HAS_TIMM:
        # include common efficientnet variants and some timm names users sometimes get
        timm_candidates += [
            "efficientnet_b0", "efficientnet_b3",
            "tf_efficientnet_b0_ns", "tf_efficientnet_b3_ns",
            "tf_efficientnet_b0", "tf_efficientnet_b3",
            "mobilenetv3_large_100", "resnet50"
        ]
        # ensure uniqueness
        timm_candidates = list(dict.fromkeys(timm_candidates))

    # Try timm candidates first
    if _HAS_TIMM:
        for cand in timm_candidates:
            tried.append(f"timm:{cand}")
            try:
                model = timm.create_model(cand, pretrained=False, num_classes=num_classes)
            except Exception:
                # candidate not available or creation failed
                continue
            ok, msg = try_load_state_dict_safely(model, state)
            if ok:
                model.to(device).eval()
                print(f"[build_model] Loaded checkpoint into timm model `{cand}`")
                return model, class_to_idx, cand
            else:
                # keep going
                # print compact mismatch message
                print(f"[build_model] Candidate `{cand}` failed to load: {msg.splitlines()[0]}")
                del model
                torch.cuda.empty_cache()

    # Try torchvision ResNet50 fallback
    tried.append("torchvision:resnet50")
    try:
        from torchvision import models
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        ok, msg = try_load_state_dict_safely(model, state)
        if ok:
            model.to(device).eval()
            print("[build_model] Loaded checkpoint into torchvision.resnet50")
            return model, class_to_idx, "resnet50"
        else:
            print("[build_model] torchvision.resnet50 failed to load exactly:", msg.splitlines()[0])
            del model
            torch.cuda.empty_cache()
    except Exception:
        pass

    # As a last resort: try a fallback (efficientnet_b0 or resnet50) with strict=False
    fallback = None
    fallback_name = None
    if _HAS_TIMM:
        try:
            fallback = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
            fallback_name = "efficientnet_b0"
        except Exception:
            fallback = None
    if fallback is None:
        try:
            from torchvision import models
            fallback = models.resnet50(pretrained=False)
            fallback.fc = torch.nn.Linear(fallback.fc.in_features, num_classes)
            fallback_name = "resnet50"
        except Exception:
            fallback = None

    if fallback is not None:
        try:
            missing_keys, unexpected_keys = fallback.load_state_dict(state, strict=False)
            print(f"[build_model] Fallback loaded with strict=False ({fallback_name}).")
            print("  number missing keys (or example):", (len(missing_keys) if isinstance(missing_keys, list) else missing_keys))
            fallback.to(device).eval()
            return fallback, class_to_idx, f"{fallback_name}_fallback_strictFalse"
        except Exception as e:
            print("[build_model] fallback strict=False failed:", e)

    # Nothing worked
    raise RuntimeError(f"[build_model] Unable to build a model compatible with checkpoint: {ckpt_path}. Tried candidates: {tried}. "
                       "If you used a custom architecture, you must supply a compatible model-building function or include 'model_name' metadata in the checkpoint.")

def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def load_image_paths(folder):
    p = Path(folder)
    files = sorted(list(p.rglob('*.jpg')) + list(p.rglob('*.jpeg')) + list(p.rglob('*.png')))
    return files

def predict_probs_for_all_images_single_model(ckpt, device, image_size, files, tta_views=3):
    """
    Loads the model, runs TTA inference for all files, returns probs ndarray shape (N, C),
    plus the class_to_idx and class_names used by the model.
    """
    print(f"[predict] Building model for checkpoint: {ckpt}")
    model, class_to_idx, model_name_used = build_model_from_ckpt(ckpt, device)
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    transform = get_transform(image_size)

    N = len(files)
    probs_list = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, fp in enumerate(files):
            img = Image.open(fp).convert('RGB')
            x = transform(img).unsqueeze(0)  # (1,C,H,W)
            per_view = []
            # original
            logits = model(x.to(device))
            per_view.append(F.softmax(logits, dim=1).cpu().numpy().squeeze())
            if tta_views >= 2:
                flipped = torch.flip(x, dims=[3])
                logits = model(flipped.to(device))
                per_view.append(F.softmax(logits, dim=1).cpu().numpy().squeeze())
            if tta_views >= 3:
                vfl = torch.flip(x, dims=[2])
                logits = model(vfl.to(device))
                per_view.append(F.softmax(logits, dim=1).cpu().numpy().squeeze())
            avg = np.mean(np.vstack(per_view), axis=0)  # (C,)
            probs_list.append(avg)
            if (i+1) % 200 == 0 or i == N-1:
                print(f"  processed {i+1}/{N} images")
    probs = np.vstack(probs_list)  # (N, C)
    # free model
    del model
    torch.cuda.empty_cache()
    return probs, class_to_idx, class_names

# -------------------------
# Main orchestration
# -------------------------
def main(args):
    device = args.device if args.device in ('cpu','cuda') else ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpts = [Path(x) for x in args.ckpts]
    files = load_image_paths(args.input)
    if not files:
        print("No images found in input folder.")
        return

    # determine per-ckpt image sizes
    if args.image_sizes:
        if len(args.image_sizes) != len(ckpts):
            raise RuntimeError("Number of --image_sizes must match number of ckpts")
        image_sizes = [int(x) for x in args.image_sizes]
    else:
        image_sizes = [infer_image_size_from_ckpt_path(p) for p in ckpts]

    print("Models and image sizes:")
    for c,s in zip(ckpts, image_sizes):
        print(" ", c, "->", s)

    # create temp folder to store per-model probs
    tmpdir = Path(tempfile.mkdtemp(prefix="ensemble_probs_"))
    per_model_files = []
    common_class_to_idx = None
    class_names = None

    print(f"Running inference sequentially on {len(ckpts)} models for {len(files)} images...")
    for i, (ckpt, img_sz) in enumerate(zip(ckpts, image_sizes)):
        print(f"[{i+1}/{len(ckpts)}] Processing model {ckpt} (image_size={img_sz}) ...")
        probs, c2i, c_names = predict_probs_for_all_images_single_model(str(ckpt), device, img_sz, files, tta_views=args.tta_views)
        pfile = tmpdir / f"probs_model_{i}.npy"
        np.save(pfile, probs)
        per_model_files.append(pfile)
        if common_class_to_idx is None:
            common_class_to_idx = c2i
            class_names = c_names
        else:
            if c2i != common_class_to_idx:
                raise RuntimeError("class_to_idx differs between checkpoints. Ensure same class mapping.")
        print("  done, saved probs ->", pfile)

    # load all per-model probs and average
    print("Averaging per-model probabilities...")
    stacked = []
    for pfile in per_model_files:
        arr = np.load(pfile)
        stacked.append(arr)
    stacked = np.stack(stacked, axis=0)  # (M, N, C)
    ensemble_probs = np.mean(stacked, axis=0)  # (N, C)

    # write CSV
    out_csv = Path(args.out_csv)
    header = ['filename','ground_truth','pred','pred_prob'] + class_names
    with out_csv.open('w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        all_true = []
        all_pred = []
        for i, fp in enumerate(files):
            probs = ensemble_probs[i]
            pred_idx = int(np.argmax(probs))
            pred_label = class_names[pred_idx]
            pred_prob = float(probs[pred_idx])
            gt = fp.parent.name if fp.parent.name in class_names else ''
            row = [str(fp), gt, pred_label, f"{pred_prob:.6f}"] + [f"{float(x):.6f}" for x in probs]
            writer.writerow(row)
            if gt != '':
                all_true.append(gt)
                all_pred.append(pred_label)

    print("Saved ensemble predictions ->", out_csv)
    if all_true:
        print("\nClassification report:")
        print(classification_report(all_true, all_pred, digits=3))
        print("\nConfusion matrix:")
        print(confusion_matrix(all_true, all_pred))

    # cleanup temp
    try:
        for f in per_model_files:
            os.remove(f)
        os.rmdir(tmpdir)
    except Exception:
        pass

# -------------------------
# CLI
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts', nargs='+', required=True, help='paths to checkpoints (space separated)')
    parser.add_argument('--input', required=True, help='input folder (recursive)')
    parser.add_argument('--out_csv', required=True, help='output CSV path')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--tta_views', type=int, default=3, choices=[1,2,3], help='1=none,2=hflip,3=h+vflip')
    parser.add_argument('--image_sizes', nargs='+', help='optional per-model image sizes (override inference), supply same count as ckpts')
    args = parser.parse_args()
    main(args)
