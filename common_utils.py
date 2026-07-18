"""
common_utils.py

Shared paths, dataset classes, and model helper definitions used by every
other script in this repo. Consolidated (not modified) from the
B.R.I.D.G.E.R. V2 Colab notebook's repeated "definitions" cells.

Import this module first in any of the other scripts, e.g.:

    from common_utils import *
"""

import os
import sys
import shutil
import zipfile
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------------
# Core paths (originally set up in the "FRESH MASTER SETUP CELL")
# ------------------------------------------------------------------
RAFDB_BASE_CACHE = '/content/drive/MyDrive/Colab Notebooks/RAF-DB/'
DRIVE_ZIP = RAFDB_BASE_CACHE + 'basic/Image/aligned.zip'
RAFDB_LABEL_FILE = RAFDB_BASE_CACHE + 'basic/EmoLabel/list_partition_label.txt'
RAFDB_IMG_DIR = '/content/rafdb_aligned/'
CSV_DIR = '/content/rafdb_csv/'
POSTER_VAR_DIR = '/content/poster_var/'
POSTER_VAR_CACHE = '/content/drive/MyDrive/Colab Notebooks/poster_var/'

CHECKPOINT_PATH = POSTER_VAR_CACHE + 'posterv2_var_rafdb_best_v3.pth'
EMA_CHECKPOINT_PATH = POSTER_VAR_CACHE + 'posterv2_var_rafdb_ema_v3.pth'

FEATURE_CACHE = POSTER_VAR_CACHE + 'features/'

NEUTRAL_CLASS = 6
N_RACE, N_GENDER, N_AGE = 3, 3, 5
GENDER_GROUPS = 3  # male, female, unsure, fixed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(d):
    return {k: v.to(device) for k, v in d.items()}


# ------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ------------------------------------------------------------------
# Dataset classes
# ------------------------------------------------------------------
class RafDbDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['filename'])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(row['label'])


class RafDbDemogDataset(Dataset):
    """Yields (image, emotion, race, gender, age) for the joint multitask model."""
    def __init__(self, csv_file, img_dir, demo_map, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.demo_map = demo_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['filename']
        img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        emo = int(row['label'])
        if fname in self.demo_map:
            d = self.demo_map[fname]
            return img, emo, d['race'], d['gender'], d['age']
        else:
            # fallback if missing, though the original check showed 0 missing
            return img, emo, -1, -1, -1


# ------------------------------------------------------------------
# Standard eval/test loaders (built once CSVs exist)
# ------------------------------------------------------------------
def build_standard_loaders(batch_size=48, num_workers=2):
    val_dataset = RafDbDataset(os.path.join(CSV_DIR, 'valid_labels.csv'), RAFDB_IMG_DIR, eval_transform)
    test_dataset = RafDbDataset(os.path.join(CSV_DIR, 'test_labels.csv'), RAFDB_IMG_DIR, eval_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return val_dataset, test_dataset, val_loader, test_loader


# ------------------------------------------------------------------
# Simple evaluation helpers
# ------------------------------------------------------------------
def evaluate_model(m, loader):
    m.eval()
    c, t = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            out = m(inputs)
            _, pred = torch.max(out.data, 1)
            t += labels.size(0)
            c += (pred == labels).sum().item()
    return c / t


def check_checkpoint(path, model_ctor, val_loader, test_loader):
    m = model_ctor().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    val_acc = evaluate_model(m, val_loader)
    test_acc = evaluate_model(m, test_loader)
    print(f"{path.split('/')[-1]}: Val {val_acc:.4f}, Test {test_acc:.4f}")
    return m, val_acc, test_acc


@torch.no_grad()
def per_group_accuracy(pred, true, n_groups):
    accs = []
    for g in range(n_groups):
        m = true == g
        if m.sum() == 0:
            accs.append(np.nan)
            continue
        accs.append((pred[m] == true[m]).float().mean().item())
    return accs


# ------------------------------------------------------------------
# Demographic classifier (unconditioned probe used throughout)
# ------------------------------------------------------------------
class DemogClf(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3))
        self.race = nn.Linear(256, N_RACE)
        self.gender = nn.Linear(256, N_GENDER)
        self.age = nn.Linear(256, N_AGE)

    def forward(self, f):
        h = self.trunk(f)
        return self.race(h), self.gender(h), self.age(h)


# Alias used in some later cells of the notebook (trunk-only classifier with
# separate named heads, functionally identical to DemogClf).
class DemogClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
        )
        self.race_head = nn.Linear(256, N_RACE)
        self.gender_head = nn.Linear(256, N_GENDER)
        self.age_head = nn.Linear(256, N_AGE)

    def forward(self, f):
        h = self.trunk(f)
        return self.race_head(h), self.gender_head(h), self.age_head(h)


def load_pyramid_model():
    """Import lazily so this module doesn't hard-require the poster-var repo
    unless a script actually needs the backbone."""
    if POSTER_VAR_DIR.rstrip('/') not in sys.path:
        sys.path.append(POSTER_VAR_DIR.rstrip('/'))
    from trails.posterv2.PosterV2_7cls import pyramid_trans_expr2
    return pyramid_trans_expr2
