# apply_thresholds_to_csv.py
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

csv_in = r"C:\Varun\skin-detect-mvp\predictions_ensemble.csv"
csv_out = r"C:\Varun\skin-detect-mvp\predictions_ensemble_thresholded.csv"
thresh_path = r"/skin-detect-mvp/ensemble_thresholds.json"

df = pd.read_csv(csv_in)
with open(thresh_path, 'r') as f:
    thresholds = json.load(f)

# class columns start at 5th column (index 4) per script header
class_cols = df.columns.tolist()[4:]
probs = df[class_cols].values
N, C = probs.shape

final_preds = []
for i in range(N):
    exceeded = []
    for ci, cname in enumerate(class_cols):
        t = thresholds.get(cname, 0.5)  # default 0.5 if missing
        if probs[i, ci] >= t:
            exceeded.append((probs[i,ci], cname))
    if len(exceeded) == 0:
        pred = class_cols[int(np.argmax(probs[i]))]
    else:
        pred = max(exceeded, key=lambda x: x[0])[1]
    final_preds.append(pred)

df['pred_thresholded'] = final_preds
df.to_csv(csv_out, index=False)
print("Saved:", csv_out)

# If ground truth exists, evaluate
if 'ground_truth' in df.columns:
    valid = df['ground_truth'].fillna('') != ''
    y_true = df.loc[valid, 'ground_truth'].values
    y_pred = df.loc[valid, 'pred_thresholded'].values
    print("\nClassification report (thresholded):")
    print(classification_report(y_true, y_pred, digits=3))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
