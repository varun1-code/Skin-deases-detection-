# export_models_torchscript.py
# Exports three models to TorchScript and writes metadata
import torch, os, json
from pathlib import Path
import timm
from torchvision import models

BASE = Path(r"C:\Varun\skin-detect-mvp")
CKPTS = {
    "b0": BASE / "outputs" / "finetune_best_epoch2_auc0.9565.pt",
    "b3": BASE / "outputs" / "best_efficientnet_b3_epoch10_auc0.9713.pt",
    "resnet50": BASE / "outputs" / "best_resnet50_epoch11_auc0.9665.pt"
}
OUT = BASE / "export"
OUT.mkdir(parents=True, exist_ok=True)

# metadata to write
metadata = {}
# thresholds file (we created it before)
thresh_path = BASE / "ensemble_thresholds.json"
if thresh_path.exists():
    with open(thresh_path, 'r') as f:
        thresholds = json.load(f)
else:
    thresholds = None

def load_state_dict_from_ckpt(ckpt_path):
    ck = torch.load(str(ckpt_path), map_location='cpu')
    state = ck.get('model_state', None)
    class_to_idx = ck.get('class_to_idx', None)
    return state, class_to_idx, ck

def build_model_for_export(kind, num_classes):
    # kind in ["b0","b3","resnet50"]
    if kind == "b3":
        m = timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)
    elif kind == "b0":
        m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    elif kind == "resnet50":
        m = models.resnet50(pretrained=False)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    else:
        raise RuntimeError("Unknown kind")
    return m

print("Exporting models to TorchScript (CPU tracing). This may take a moment...")

for kind, ckpt in CKPTS.items():
    if not ckpt.exists():
        print(f"Checkpoint missing: {ckpt}. Skipping {kind}.")
        continue
    state, class_to_idx, ck = load_state_dict_from_ckpt(ckpt)
    if class_to_idx is None:
        print(f"Warning: ckpt {ckpt} missing class_to_idx; attempting anyway.")
        # try to infer classes from existing files
        class_to_idx = ck.get('class_to_idx', None)
    num_classes = len(class_to_idx) if class_to_idx else 7

    model = build_model_for_export(kind, num_classes)
    # Attempt exact load first
    try:
        model.load_state_dict(state)
        print(f"Loaded state for {kind} exactly.")
    except Exception as e:
        print(f"Exact load failed for {kind}, trying strict=False (partial load). Error: {e}")
        model.load_state_dict(state, strict=False)

    model.eval()
    # Create a dummy input with appropriate size
    img_size = 300 if kind == "b3" else 224
    dummy = torch.randn(1,3,img_size,img_size)

    # Convert to TorchScript using scripting (preferred for control-flow-safe)
    try:
        scripted = torch.jit.trace(model, dummy, strict=False)
    except Exception as e:
        print(f"Trace failed for {kind}, trying scripting. Error: {e}")
        scripted = torch.jit.script(model)

    out_path = OUT / f"model_{kind}.pt"
    scripted.save(str(out_path))
    print(f"Saved TorchScript {out_path}")

    # add metadata entry
    metadata[kind] = {
        "ckpt": str(ckpt),
        "torchscript": str(out_path),
        "image_size": img_size,
        "num_classes": num_classes
    }

# save meta-learner (already saved earlier)
meta_src = BASE / "per_model_probs" / "meta_learner.joblib"
if meta_src.exists():
    meta_dst = OUT / "meta_learner.joblib"
    import shutil
    shutil.copy2(meta_src, meta_dst)
    print("Copied meta-learner to export folder.")
else:
    print("Meta-learner joblib not found at:", meta_src)

# write thresholds and class mapping
metadata["thresholds"] = thresholds
# class names (try to read from per_model_probs)
class_json = BASE / "per_model_probs" / "class_names.json"
if class_json.exists():
    with open(class_json, 'r') as f:
        metadata["class_names"] = json.load(f)
else:
    metadata["class_names"] = None

with open(OUT / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print("Export complete. Files written to:", OUT)
print("Metadata written to:", OUT / "metadata.json")
