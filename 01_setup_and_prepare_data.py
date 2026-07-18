"""
01_setup_and_prepare_data.py

Covers:
  1. Mount Drive, download the poster-var repo and IR50/MobileFaceNet weights
  2. Patch hardcoded Windows paths in the repo
  3. Copy RAF-DB images locally and build train/val/test label CSVs
  4. Verify / restore the POSTER-Var checkpoints from `new_saves`
  5. Parse the RAF-DB manual demographic annotations (gender/race/age)
  6. Attach those demographics to the cached 768-d POSTER-Var features
"""

import os
import shutil
import zipfile
import requests

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from common_utils import (
    RAFDB_BASE_CACHE, DRIVE_ZIP, RAFDB_LABEL_FILE, RAFDB_IMG_DIR, CSV_DIR,
    POSTER_VAR_DIR, POSTER_VAR_CACHE, CHECKPOINT_PATH, EMA_CHECKPOINT_PATH,
    FEATURE_CACHE, device, build_standard_loaders, check_checkpoint,
    load_pyramid_model,
)


# ============================================================
# 1. Master setup: repo, weights, images, CSVs
# ============================================================
def master_setup():
    import gdown
    from google.colab import drive
    drive.mount('/content/drive')

    # Download poster-var repo
    if not os.path.exists(os.path.join(POSTER_VAR_DIR, 'trails', 'posterv2')):
        print("Downloading poster-var repo...")
        os.makedirs(POSTER_VAR_DIR, exist_ok=True)
        r = requests.get("https://github.com/lg2578/poster-var/archive/refs/heads/main.zip")
        with open('/content/poster_var.zip', 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile('/content/poster_var.zip', 'r') as z:
            z.extractall('/content/poster_var_tmp')
        ef = os.listdir('/content/poster_var_tmp')[0]
        for item in os.listdir(f'/content/poster_var_tmp/{ef}'):
            shutil.move(f'/content/poster_var_tmp/{ef}/{item}', f'{POSTER_VAR_DIR}{item}')
        shutil.rmtree('/content/poster_var_tmp')
        print("Downloaded.")
    else:
        print("poster-var repo already present.")

    import sys
    if POSTER_VAR_DIR.rstrip('/') not in sys.path:
        sys.path.append(POSTER_VAR_DIR.rstrip('/'))

    # IR50 weights
    ir50_path = f'{POSTER_VAR_DIR}models/pretrain/ir50.pth'
    if not os.path.exists(ir50_path):
        print("Downloading IR50 weights...")
        os.makedirs(os.path.dirname(ir50_path), exist_ok=True)
        gdown.download(id="17QAIPlpZUwkQzOTNiu-gUFLTqAxS-qHt", output=ir50_path)
    else:
        print("IR50 weights already present.")

    mobilefacenet_path = f'{POSTER_VAR_DIR}models/mobilefacenet_model_best.pth.tar'
    print("MobileFaceNet present:", os.path.exists(mobilefacenet_path))

    # Patch hardcoded Windows paths
    posterv2_file = f'{POSTER_VAR_DIR}trails/posterv2/PosterV2_7cls.py'
    with open(posterv2_file, 'r') as f:
        content = f.read()
    if r'D:\lg\models' in content:
        content = content.replace(
            r"r'D:\lg\models\mobilefacenet_model_best.pth.tar'", f"'{mobilefacenet_path}'")
        content = content.replace(r"r'D:\lg\models\ir50.pth'", f"'{ir50_path}'")
        with open(posterv2_file, 'w') as f:
            f.write(content)
        print("Patched paths.")
    else:
        print("Already patched.")

    # Copy RAF-DB images via zip
    if not os.path.exists(RAFDB_IMG_DIR) or len(os.listdir(RAFDB_IMG_DIR)) < 1:
        print("Copying RAF-DB images via zip...")
        if os.path.exists(RAFDB_IMG_DIR):
            shutil.rmtree(RAFDB_IMG_DIR)
        shutil.copy(DRIVE_ZIP, '/content/aligned.zip')
        with zipfile.ZipFile('/content/aligned.zip', 'r') as z:
            z.extractall('/content/rafdb_aligned_tmp')
        items = os.listdir('/content/rafdb_aligned_tmp')
        if len(items) == 1 and os.path.isdir(f'/content/rafdb_aligned_tmp/{items[0]}'):
            shutil.move(f'/content/rafdb_aligned_tmp/{items[0]}', RAFDB_IMG_DIR)
            shutil.rmtree('/content/rafdb_aligned_tmp')
        else:
            shutil.move('/content/rafdb_aligned_tmp', RAFDB_IMG_DIR)
        print("Copied.")
    else:
        print("RAF-DB images already local.")
    print("Image count:", len(os.listdir(RAFDB_IMG_DIR)))

    # Build CSVs
    if not os.path.exists(os.path.join(CSV_DIR, 'train_labels.csv')):
        print("Building CSVs...")
        labels_df = pd.read_csv(RAFDB_LABEL_FILE, sep=' ', header=None, names=['filename', 'label'])
        labels_df['label'] = labels_df['label'] - 1
        labels_df['filename'] = labels_df['filename'].apply(
            lambda n: f"{os.path.splitext(n)[0]}_aligned.jpg")
        train_df = labels_df[labels_df['filename'].str.startswith('train')]
        test_df = labels_df[labels_df['filename'].str.startswith('test')].reset_index(drop=True)
        tr, va = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=0)
        os.makedirs(CSV_DIR, exist_ok=True)
        tr.to_csv(os.path.join(CSV_DIR, 'train_labels.csv'), index=False)
        va.to_csv(os.path.join(CSV_DIR, 'valid_labels.csv'), index=False)
        test_df.to_csv(os.path.join(CSV_DIR, 'test_labels.csv'), index=False)
        print(f"Built. Train: {len(tr)}, Val: {len(va)}, Test: {len(test_df)}")
    else:
        print("CSVs already built.")

    print("\nChecking checkpoint files:")
    for p in [CHECKPOINT_PATH, EMA_CHECKPOINT_PATH]:
        exists = os.path.exists(p)
        print(f"  {p.split('/')[-1]}: {'present' if exists else 'MISSING'}")
    print("\nSetup complete.")


# ============================================================
# 2. Verify checkpoints and restore clean copies from new_saves
# ============================================================
def verify_and_restore_checkpoints():
    pyramid_trans_expr2 = load_pyramid_model()
    model_ctor = lambda: pyramid_trans_expr2(img_size=224, num_classes=7, vae=True)
    _, _, val_loader, test_loader = build_standard_loaders()

    new_saves_dir = '/content/drive/MyDrive/new_saves/'
    saved_raw = new_saves_dir + 'posterv2_var_rafdb_best_v3.pth'
    saved_ema = new_saves_dir + 'posterv2_var_rafdb_ema_v3.pth'

    print("Raw best (from new_saves):")
    check_checkpoint(saved_raw, model_ctor, val_loader, test_loader)
    print("\nEMA (from new_saves):")
    check_checkpoint(saved_ema, model_ctor, val_loader, test_loader)

    for fname in ['grouping_results_20seed.csv', 'sparsity_sweep_10seed.csv', 'demog_decodability_10seed.csv']:
        path = POSTER_VAR_CACHE + fname
        exists = os.path.exists(path)
        print(f"{fname}: {'present' if exists else 'MISSING'}")
        if exists:
            df = pd.read_csv(path)
            print(f"  {len(df)} rows")

    shutil.copy(saved_raw, CHECKPOINT_PATH)
    shutil.copy(saved_ema, EMA_CHECKPOINT_PATH)
    print("Overwrote corrupted checkpoints with clean ones. Re-verifying:")
    check_checkpoint(CHECKPOINT_PATH, model_ctor, val_loader, test_loader)
    check_checkpoint(EMA_CHECKPOINT_PATH, model_ctor, val_loader, test_loader)


# ============================================================
# 3. Parse RAF-DB manual demographic annotations
# ============================================================
def parse_attri(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    gender = int(lines[5])
    race = int(lines[6])
    age = int(lines[7])
    return gender, race, age


def inspect_manual_annotations():
    manual1_dir = RAFDB_BASE_CACHE + 'basic/Annotation/manual (1)/'
    print("manual (1) exists:", os.path.exists(manual1_dir))
    if os.path.exists(manual1_dir):
        files = sorted(os.listdir(manual1_dir))
        print("File count:", len(files))
        print("First few:", files[:3])
        for fname in files[:2]:
            print(f"\n--- {fname} ---")
            with open(os.path.join(manual1_dir, fname)) as f:
                print(f.read())


def parse_demographics_local_copy():
    """Copy the annotation folder locally first (avoids slow one-by-one Drive
    reads), then parse every file: 5 landmark lines, then gender/race/age."""
    local_manual_dir = '/content/rafdb_manual_annotations/'
    if not os.path.exists(local_manual_dir):
        print("Copying annotation folder to local disk...")
        shutil.copytree(RAFDB_BASE_CACHE + 'basic/Annotation/manual (1)/', local_manual_dir)
        print("Copied.")
    else:
        print("Already local.")
    print("File count:", len(os.listdir(local_manual_dir)))

    records = {}
    files = [f for f in os.listdir(local_manual_dir) if f.endswith('_manu_attri.txt')]
    for i, fname in enumerate(files):
        if i % 2000 == 0:
            print(f"{i}/{len(files)} processed...")
        img_id = fname.replace('_manu_attri.txt', '')
        g, r, a = parse_attri(os.path.join(local_manual_dir, fname))
        records[f"{img_id}_aligned.jpg"] = {'gender': g, 'race': r, 'age': a}

    demo_df = pd.DataFrame.from_dict(records, orient='index')
    print("\nTotal parsed:", len(demo_df))
    print("\nGender counts (0=male,1=female,2=unsure):")
    print(demo_df['gender'].value_counts().sort_index())
    print("\nRace counts (0=White,1=Black,2=Asian):")
    print(demo_df['race'].value_counts().sort_index())
    print("\nAge counts (bins):")
    print(demo_df['age'].value_counts().sort_index())
    return demo_df


def parse_demographics_and_save():
    """Parse directly from Drive (no local copy), split by train/test prefix,
    and save the aligned demographics CSV used by every downstream script."""
    manual1_dir = RAFDB_BASE_CACHE + 'basic/Annotation/manual (1)/'
    records = {}
    for fname in os.listdir(manual1_dir):
        if not fname.endswith('_manu_attri.txt'):
            continue
        img_id = fname.replace('_manu_attri.txt', '')
        g, r, a = parse_attri(os.path.join(manual1_dir, fname))
        records[f"{img_id}_aligned.jpg"] = {'gender': g, 'race': r, 'age': a}

    demo_df = pd.DataFrame.from_dict(records, orient='index')
    demo_df.index.name = 'filename'
    demo_df = demo_df.reset_index()
    demo_df['split'] = demo_df['filename'].apply(lambda x: 'test' if x.startswith('test') else 'train')

    print("Age distribution by split:")
    print(pd.crosstab(demo_df['age'], demo_df['split']))
    print("\nRace distribution by split:")
    print(pd.crosstab(demo_df['race'], demo_df['split']))
    print("\nGender distribution by split:")
    print(pd.crosstab(demo_df['gender'], demo_df['split']))

    demo_df.to_csv(POSTER_VAR_CACHE + 'rafdb_demographics.csv', index=False)
    print("\nSaved demographics to:", POSTER_VAR_CACHE + 'rafdb_demographics.csv')
    return demo_df


# ============================================================
# 4. Attach demographics to cached POSTER-Var features
# ============================================================
def attach_demographics(split):
    demo_df = pd.read_csv(POSTER_VAR_CACHE + 'rafdb_demographics.csv')
    demo_map = demo_df.set_index('filename')[['race', 'gender', 'age']].to_dict('index')

    csv_path = os.path.join(CSV_DIR, f"{'valid' if split == 'val' else split}_labels.csv")
    order_df = pd.read_csv(csv_path)
    cache = torch.load(FEATURE_CACHE + f'posterv2_ema_{split}.pt')
    feats, emo_labels = cache['features'], cache['labels']

    assert len(order_df) == feats.shape[0], f"{split}: {len(order_df)} vs {feats.shape[0]}"

    races, genders, ages = [], [], []
    missing = 0
    for fname in order_df['filename']:
        if fname in demo_map:
            d = demo_map[fname]
            races.append(d['race']); genders.append(d['gender']); ages.append(d['age'])
        else:
            races.append(-1); genders.append(-1); ages.append(-1); missing += 1

    out = {
        'features': feats,
        'emotion': emo_labels,
        'race': torch.tensor(races),
        'gender': torch.tensor(genders),
        'age': torch.tensor(ages),
    }
    torch.save(out, FEATURE_CACHE + f'posterv2_ema_{split}_full.pt')
    print(f"{split}: {feats.shape[0]} samples, {missing} missing demographics")
    return out


def attach_demographics_all_splits():
    for split in ['train', 'val', 'test']:
        attach_demographics(split)
    print("\nDemographics aligned to cached features and saved as *_full.pt")


if __name__ == '__main__':
    master_setup()
    verify_and_restore_checkpoints()
    inspect_manual_annotations()
    parse_demographics_and_save()
    attach_demographics_all_splits()
