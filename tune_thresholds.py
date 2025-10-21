# tune_thresholds.py
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

csv_path = r"C:\Varun\skin-detect-mvp\predictions_ensemble.csv"
df = pd.read_csv(csv_path)

# find class columns (after 4th column)
cols = df.columns.tolist()
# header is: filename, ground_truth, pred, pred_prob, <class names...>
class_cols = cols[4:]
class_names = class_cols

y_true = df['ground_truth'].fillna('').values
# build binary one-hot ground truth matrix (unknown gt -> skip)
valid_mask = (y_true != '')
y_true_valid = y_true[valid_mask]
N = len(df)
probs = df[class_cols].values  # shape N x C

# Build true one-hot for valid rows
C = probs.shape[1]
y_onehot = np.zeros((N, C), dtype=int)
for i, gt in enumerate(y_true):
    if gt != '':
        try:
            idx = class_names.index(gt)
            y_onehot[i, idx] = 1
        except ValueError:
            pass

# Optimize thresholds per class (use only images with ground-truth)
best_thresh = {}
for ci, cname in enumerate(class_names):
    best_f1 = -1.0
    best_t = 0.5
    # thresholds from 0.01 to 0.99
    for t in np.linspace(0.01, 0.99, 99):
        preds_bin = (probs[:, ci] >= t).astype(int)
        # evaluate only where GT exists
        if valid_mask.sum() == 0:
            continue
        f1 = f1_score(y_onehot[valid_mask, ci], preds_bin[valid_mask], zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_t = t
    best_thresh[cname] = (best_t, best_f1)

print("Best thresholds (per-class):")
for k,(t,f) in best_thresh.items():
    print(f"  {k}: thresh={t:.2f}, f1={f:.3f}")

# Apply thresholds to make final predictions:
final_preds = []
for i in range(N):
    # check which classes exceed threshold
    exceeds = []
    for ci, cname in enumerate(class_names):
        if probs[i,ci] >= best_thresh[cname][0]:
            exceeds.append((probs[i,ci], cname))
    if len(exceeds) == 0:
        # fallback: choose argmax
        chosen = class_names[int(np.argmax(probs[i]))]
    else:
        # if multiple, pick class with highest prob among them
        chosen = max(exceeds, key=lambda x: x[0])[1]
    final_preds.append(chosen)

# Evaluate on images with GT
gt_list = list(y_true[valid_mask])
pred_list = [final_preds[i] for i in range(N) if valid_mask[i]]

print("\nClassification report after thresholding:")
print(classification_report(gt_list, pred_list, digits=3))
print("\nConfusion matrix (after thresholding):")
print(confusion_matrix(gt_list, pred_list))
