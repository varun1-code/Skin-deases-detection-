# save_per_model_probs_for_stacking.py
# Usage: edit the CKPTS list below or pass via command line if you prefer.
import torch, os, json, numpy as np
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

# try to import timm
try:
    import timm
    _HAS_TIMM = True
except Exception:
    timm = None
    _HAS_TIMM = False

from torchvision import transforms, models

# ---------- CONFIG ----------
CKPTS = [
    r"C:\Varun\skin-detect-mvp\outputs\finetune_best_epoch2_auc0.9565.pt",
    r"C:\Varun\skin-detect-mvp\outputs\best_efficientnet_b3_epoch10_auc0.9713.pt",
    r"C:\Varun\skin-detect-mvp\outputs\best_resnet50_epoch11_auc0.9665.pt"
]
DATA_DIR = Path(r"C:\Varun\skin-detect-mvp\data")
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
OUT_DIR = Path(r"C:\Varun\skin-detect-mvp\per_model_probs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TTA_VIEWS = 1   # set 1 for faster; we'll keep stacking training simple (use original probs)
# ----------------------------

def infer_image_size(path_str):
    s = path_str.lower()
    if 'b3' in s or 'efficientnet_b3' in s:
        return 300
    return 224

def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize(int(image_size*1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def build_model_from_ckpt(ckpt_path, num_classes):
    ck = torch.load(ckpt_path, map_location='cpu')
    state = ck.get('model_state', ck)
    # try timm candidates
    tried = []
    if _HAS_TIMM:
        for cand in ["efficientnet_b0","efficientnet_b3","tf_efficientnet_b0","tf_efficientnet_b3","resnet50"]:
            tried.append(("timm", cand))
            try:
                m = timm.create_model(cand, pretrained=False, num_classes=num_classes)
            except Exception:
                continue
            try:
                m.load_state_dict(state)
                return m
            except Exception:
                del m
    # fallback torchvision resnet50
    try:
        m = models.resnet50(pretrained=False)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        m.load_state_dict(state)
        return m
    except Exception:
        pass
    # last-resort: attempt timm efficientnet_b0 with strict=False
    if _HAS_TIMM:
        try:
            m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
            m.load_state_dict(state, strict=False)
            return m
        except Exception:
            pass
    raise RuntimeError(f"Could not build model for checkpoint {ckpt_path} (tried {tried})")

def list_images(folder: Path):
    imgs = sorted(list(folder.rglob("*.jpg")) + list(folder.rglob("*.jpeg")) + list(folder.rglob("*.png")))
    return imgs

# load class mapping from first checkpoint
first_ck = torch.load(CKPTS[0], map_location='cpu')
class_to_idx = first_ck.get('class_to_idx', None)
if class_to_idx is None:
    raise RuntimeError("First checkpoint missing class_to_idx. Can't proceed.")
idx_to_class = {v:k for k,v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

# save class names
with open(OUT_DIR / "class_names.json","w") as f:
    json.dump(class_names, f, indent=2)

# prepare file lists
val_files = list_images(VAL_DIR)
test_files = list_images(TEST_DIR)
with open(OUT_DIR / "filenames_val.txt","w",encoding="utf8") as f:
    f.write("\n".join([str(p) for p in val_files]))
with open(OUT_DIR / "filenames_test.txt","w",encoding="utf8") as f:
    f.write("\n".join([str(p) for p in test_files]))

print(f"VAL images: {len(val_files)}, TEST images: {len(test_files)}")
print("Device:", DEVICE)

for ck in CKPTS:
    stem = Path(ck).stem
    print("\nProcessing checkpoint:", ck)
    img_size = infer_image_size(ck)
    print("  inferred image size:", img_size)
    model = build_model_from_ckpt(ck, num_classes=len(class_names))
    model.to(DEVICE).eval()
    transform = get_transform(img_size)

    def infer_list(files, out_path):
        probs_list = []
        with torch.no_grad():
            for i,p in enumerate(files):
                img = Image.open(p).convert("RGB")
                x = transform(img).unsqueeze(0).to(DEVICE)
                # basic TTA if desired
                logits = model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
                probs_list.append(probs)
                if (i+1) % 200 == 0 or i == len(files)-1:
                    print(f"   processed {i+1}/{len(files)}")
        arr = np.vstack(probs_list)
        np.save(out_path, arr)
        print("  saved:", out_path)

    out_val = OUT_DIR / f"{stem}_val.npy"
    out_test = OUT_DIR / f"{stem}_test.npy"
    infer_list(val_files, out_val)
    infer_list(test_files, out_test)

print("\nAll done. Per-model probs saved to:", OUT_DIR)
