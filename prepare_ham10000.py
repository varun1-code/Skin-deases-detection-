# prepare_ham10000.py
import zipfile
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from collections import Counter

# ----- USER PATHS (already in your environment) -----
zip_paths = [
    r"C:\Varun\skin-detect-mvp\HAM10000_images_part_1.zip",
    r"C:\Varun\skin-detect-mvp\HAM10000_images_part_2.zip",
]
meta_path = r"C:\Varun\skin-detect-mvp\HAM10000_metadata.csv"
out_images_dir = r"C:\Varun\skin-detect-mvp\images"   # extracted images here
out_data_dir = r"C:\Varun\skin-detect-mvp\data"       # final split -> data/train, data/val, data/test

os.makedirs(out_images_dir, exist_ok=True)
os.makedirs(out_data_dir, exist_ok=True)

# ----- 1) Extract zip files (if image already exists, skip) -----
print("Extracting zip files...")
for zpath in zip_paths:
    print("  ->", zpath)
    with zipfile.ZipFile(zpath, 'r') as zf:
        for member in zf.namelist():
            # skip directories
            if member.endswith('/'):
                continue
            target = os.path.join(out_images_dir, os.path.basename(member))
            if not os.path.exists(target):
                try:
                    zf.extract(member, out_images_dir)
                    # if extracted into nested path, move to flat folder
                    extracted = os.path.join(out_images_dir, member)
                    if extracted != target and os.path.exists(extracted):
                        os.replace(extracted, target)
                        # remove empty dirs if any
                        extracted_dir = os.path.dirname(extracted)
                        try:
                            os.removedirs(extracted_dir)
                        except Exception:
                            pass
                except Exception as e:
                    print("    extraction error for", member, e)
            else:
                pass  # already exists
print("Extraction done. Images directory:", out_images_dir)

# ----- 2) Read metadata CSV -----
print("\nReading metadata...")
df = pd.read_csv(meta_path)
print("Metadata rows:", len(df))
print("Metadata columns:", df.columns.tolist())

# Common HAM10000 metadata has 'image_id' and 'dx' columns
if 'image_id' not in df.columns or 'dx' not in df.columns:
    raise SystemExit("Expected metadata columns 'image_id' and 'dx' not found. Inspect metadata CSV.")

# Map image filenames -> dx
df['filename'] = df['image_id'].astype(str) + '.jpg'
# Filter only files that exist in extracted folder
df['exists'] = df['filename'].apply(lambda f: os.path.exists(os.path.join(out_images_dir, f)))
missing_files = df[~df['exists']]
if len(missing_files) > 0:
    print(f"Warning: {len(missing_files)} images referenced in metadata are missing from the extracted folder.")
    print("Example missing filenames:", missing_files['filename'].head().tolist())
# Keep only existing files for splitting
df = df[df['exists']].copy()
print("Usable image rows after existence check:", len(df))

# ----- 3) Per-class counts -----
counts = df['dx'].value_counts()
print("\nPer-class counts (usable images):")
print(counts.to_string())

# ----- 4) Stratified split: train 70%, val 15%, test 15% -----
print("\nCreating stratified splits (train 70% / val 15% / test 15%)...")
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['dx'], random_state=42)
val_df, test_df  = train_test_split(temp_df, test_size=0.50, stratify=temp_df['dx'], random_state=42)

print("Split sizes -> train:", len(train_df), "val:", len(val_df), "test:", len(test_df))

# ----- 5) Copy files into data/<split>/<dx>/ directories -----
def copy_split(split_df, split_name):
    for dx, group in split_df.groupby('dx'):
        target_dir = os.path.join(out_data_dir, split_name, dx)
        os.makedirs(target_dir, exist_ok=True)
        for fname in group['filename']:
            src = os.path.join(out_images_dir, fname)
            dst = os.path.join(target_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    print(f"Copied {len(split_df)} files to {split_name}/ (by class folders).")

copy_split(train_df, 'train')
copy_split(val_df, 'val')
copy_split(test_df, 'test')

# ----- 6) Summary & class weights hint -----
print("\nDone. Summary of final splits (per-class counts):")
for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    print(f"\n{split_name.upper()} counts:")
    print(split_df['dx'].value_counts().to_string())

# Compute basic class weights (useful for training)
total_train = len(train_df)
class_counts = train_df['dx'].value_counts().to_dict()
class_weights = {cls: total_train / (len(class_counts) * cnt) for cls, cnt in class_counts.items()}
print("\nSuggested class weights (dictionary):")
print(class_weights)

print("\nPrepared data folder:", out_data_dir)
print("You can now point your training script to data/train, data/val, data/test.")
