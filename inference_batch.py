# inference_batch.py
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import csv
import sys
import json
# load once at startup
with open(r"/skin-detect-mvp/frontend/ensemble_thresholds.json") as f:
    THRESH = json.load(f)

def apply_thresholds_to_probs(probs_row, class_names, thresholds=THRESH):
    # probs_row: 1D numpy array of length C
    exceeded = [(p, name) for p,name in zip(probs_row, class_names) if p >= thresholds.get(name, 0.5)]
    if not exceeded:
        return class_names[int(probs_row.argmax())], float(probs_row.max())
    chosen = max(exceeded, key=lambda x: x[0])
    return chosen[1], float(chosen[0])

def load_model(checkpoint_path, device='cpu'):
    ck = torch.load(checkpoint_path, map_location=device)
    model_state = ck['model_state']
    class_to_idx = ck.get('class_to_idx', None)
    if class_to_idx is None:
        raise RuntimeError("Checkpoint missing class_to_idx mapping.")
    # Build model same as training: try timm efb0, fallback resnet50
    try:
        import timm
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(class_to_idx))
    except Exception:
        from torchvision import models
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
    model.load_state_dict(model_state)
    model.to(device).eval()
    return model, class_to_idx

def predict_image(img_path, model, device, image_size=224):
    tfm = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
    return probs

def main(args):
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    model, class_to_idx = load_model(args.ckpt, device=device)
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    out_csv = Path(args.out_csv)
    src = Path(args.input)
    files = list(src.rglob('*.jpg')) + list(src.rglob('*.jpeg')) + list(src.rglob('*.png'))
    if len(files) == 0:
        print("No images found in", src)
        sys.exit(1)

    with out_csv.open('w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        # header: filename, ground_truth (inferred from parent folder), pred, pred_prob, then per-class probs
        header = ['filename', 'ground_truth', 'pred', 'pred_prob'] + class_names
        writer.writerow(header)
        for p in files:
            probs = predict_image(p, model, device, image_size=args.image_size)
            pred_idx = int(probs.argmax())
            pred_label = class_names[pred_idx]
            pred_prob = float(probs[pred_idx])
            # infer ground truth from parent folder name if folder structure is .../<label>/image.jpg
            gt = p.parent.name if p.parent.name in class_names else ''
            row = [str(p), gt, pred_label, f"{pred_prob:.6f}"] + [f"{float(x):.6f}" for x in probs]
            writer.writerow(row)
    print(f"Saved predictions for {len(files)} images to {out_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='path to model checkpoint (.pt)')
    parser.add_argument('--input', required=True, help='input folder (will search recursively)')
    parser.add_argument('--out_csv', default='predictions.csv', help='output CSV path')
    parser.add_argument('--device', default=None, help='cuda or cpu (default: auto)')
    parser.add_argument('--image_size', type=int, default=224)
    args = parser.parse_args()
    main(args)
