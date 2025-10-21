# train_resnet50.py
"""
Train ResNet50 on HAM10000.
Defaults: image_size=224, batch_size=8.
Usage:
  python train_resnet50.py --data_dir "C:\Varun\skin-detect-mvp\data" --epochs 12 --batch_size 8 --image_size 224
"""
import argparse, os, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from torch.amp import GradScaler, autocast

try:
    import timm
    _HAS_TIMM = True
except Exception:
    timm = None
    _HAS_TIMM = False

# ---------- Config ----------
class Config:
    data_dir = r"C:\Varun\skin-detect-mvp\data"
    out_dir = r"C:\Varun\skin-detect-mvp\outputs"
    model_name = "resnet50"
    image_size = 224
    batch_size = 8
    num_workers = 4
    epochs = 12
    lr = 1e-4
    weight_decay = 1e-5
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Utils ----------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = torch.tensor(alpha, dtype=torch.float) if alpha is not None else None
    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        pt = probs[range(len(targets)), targets]
        log_pt = torch.log(torch.clamp(pt, 1e-9, 1.0))
        loss = - (1 - pt) ** self.gamma * log_pt
        if self.alpha is not None:
            at = self.alpha.to(inputs.device)[targets]
            loss = at * loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

def create_dataloaders(data_dir, image_size, batch_size, num_workers=4):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(int(image_size*1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)
    test_ds = datasets.ImageFolder(test_dir, transform=val_tfms)
    class_to_idx = train_ds.class_to_idx

    targets = [s[1] for s in train_ds.samples]
    counts = np.bincount(targets)
    class_weights = {i: 1.0/counts[i] if counts[i]>0 else 0.0 for i in range(len(counts))}
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, class_to_idx

def build_model(num_classes, model_name='resnet50', pretrained=True):
    if _HAS_TIMM and model_name in timm.list_models() and 'resnet' not in model_name:
        # prefer torchvision for resnet50
        pass
    from torchvision import models
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def compute_class_weights(train_loader):
    counts = defaultdict(int)
    for _, label in train_loader.dataset.samples:
        counts[label] += 1
    num_classes = len(train_loader.dataset.classes)
    total = sum(counts.values())
    weights = [total/(num_classes*counts[i]) if counts[i]>0 else 0.0 for i in range(num_classes)]
    return np.array(weights)

# ---------- Training / Eval ----------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train(); running_loss=0.0; n=0
    for imgs, labels in loader:
        imgs = imgs.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        if device.startswith('cuda'):
            with autocast("cuda"):
                preds = model(imgs); loss = criterion(preds, labels)
        else:
            preds = model(imgs); loss = criterion(preds, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        running_loss += loss.item() * imgs.size(0); n += imgs.size(0)
    return running_loss / n

def evaluate(model, loader, device, num_classes, class_names, use_tta=False):
    model.eval()
    all_probs=[]; all_labels=[]
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device); labels = labels.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs); all_labels.append(labels.cpu().numpy())
    all_probs = np.vstack(all_probs); all_labels = np.concatenate(all_labels)
    per_class_auc = {}
    y_true_onehot = np.zeros((len(all_labels), num_classes))
    for i, lab in enumerate(all_labels): y_true_onehot[i, lab]=1
    for c in range(num_classes):
        try: per_class_auc[class_names[c]] = roc_auc_score(y_true_onehot[:,c], all_probs[:,c])
        except: per_class_auc[class_names[c]] = np.nan
    macro_auc = np.nanmean(list(per_class_auc.values()))
    preds = np.argmax(all_probs, axis=1)
    cm = confusion_matrix(all_labels, preds)
    cls_report = classification_report(all_labels, preds, target_names=class_names, zero_division=0)
    return {'per_class_auc': per_class_auc, 'macro_auc': macro_auc, 'confusion_matrix': cm, 'classification_report': cls_report}

# ---------- Main ----------
def main(args):
    cfg = Config()
    cfg.data_dir = args.data_dir or cfg.data_dir
    cfg.epochs = args.epochs or cfg.epochs
    cfg.batch_size = args.batch_size or cfg.batch_size
    cfg.image_size = args.image_size or cfg.image_size
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    print("Device:", cfg.device)
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(cfg.data_dir, cfg.image_size, cfg.batch_size, cfg.num_workers)
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)
    print("Classes:", class_names)

    model = build_model(num_classes, model_name=cfg.model_name, pretrained=True).to(cfg.device)

    cw_np = compute_class_weights(train_loader)
    alpha = cw_np / cw_np.sum() * len(cw_np)
    print("Alpha for FocalLoss:", alpha)

    criterion = FocalLoss(alpha=alpha, gamma=2.0, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler()

    best_val_auc = -1.0; best_path = None
    for epoch in range(1, cfg.epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device, scaler)
        val_metrics = evaluate(model, val_loader, cfg.device, num_classes, class_names)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{cfg.epochs} — train_loss: {train_loss:.4f} — val_macro_auc: {val_metrics['macro_auc']:.4f} — time: {elapsed:.1f}s")
        for k,v in val_metrics['per_class_auc'].items(): print(f"  {k}: {v:.4f}")
        if not np.isnan(val_metrics['macro_auc']) and val_metrics['macro_auc'] > best_val_auc:
            best_val_auc = val_metrics['macro_auc']
            best_path = os.path.join(cfg.out_dir, f"best_{cfg.model_name}_epoch{epoch}_auc{best_val_auc:.4f}.pt")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'class_to_idx': class_to_idx}, best_path)
            print("Saved best model ->", best_path)
        if epoch % 5 == 0:
            ckpt = os.path.join(cfg.out_dir, f"ckpt_{cfg.model_name}_epoch{epoch}.pt")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'class_to_idx': class_to_idx}, ckpt)

    print("Training complete. Best val macro AUC:", best_val_auc)
    if best_path is not None:
        print("Loading best model for test evaluation:", best_path)
        chk = torch.load(best_path, map_location=cfg.device)
        model.load_state_dict(chk['model_state'])
        test_metrics = evaluate(model, test_loader, cfg.device, num_classes, class_names)
        print("TEST macro AUC:", test_metrics['macro_auc'])
        print("Test classification report:\n", test_metrics['classification_report'])
        print("Confusion matrix:\n", test_metrics['confusion_matrix'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--image_size', type=int, default=None)
    args = parser.parse_args()
    main(args)
