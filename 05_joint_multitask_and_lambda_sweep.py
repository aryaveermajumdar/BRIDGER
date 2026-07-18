"""
05_joint_multitask_and_lambda_sweep.py

Consolidates the backbone-level fix: training POSTER-Var jointly with an
auxiliary demographic-classification loss (weighted by lambda), then
re-running the demographic-conditioning comparison on the resulting
"recovered" features.

Covers:
  1. RafDbDemogDataset loaders + JointModel (emotion head + auxiliary
     race/gender/age heads off the same 768-d feature).
  2. run_lambda_sweep(): trains JointModel end-to-end (backbone frozen at
     ir_back) for several lambda values, tracking val emotion/race accuracy.
  3. Evaluation of the saved lambda=0.1 checkpoint on the test set, overall
     and per-race-group.
  4. Re-extraction of the 768-d feature from the lambda=0.1 model and
     re-attaching demographics (train/val/test).
  5. The "none" vs "hard_fine" conditioning comparison on top of the
     lambda=0.1 (recovered) features, with a full rigorous statistical
     battery (Shapiro, paired t-test, Wilcoxon, permutation test, bootstrap
     CI, Holm-corrected p-values across all three tests).
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import stats

from common_utils import (
    RAFDB_IMG_DIR, CSV_DIR, POSTER_VAR_CACHE, FEATURE_CACHE, EMA_CHECKPOINT_PATH,
    NEUTRAL_CLASS, N_RACE, N_GENDER, N_AGE, device, eval_transform, train_transform,
    RafDbDataset, RafDbDemogDataset, load_pyramid_model,
)

# these come from 04_grouping_and_shrinkage_experiments.py (Head, train_one,
# compute_metrics use the same 'none' / 'hard_fine' / 'hard_best' / 'shrinkage'
# machinery — reused here rather than redefined).
from importlib import import_module
_grouping = import_module('04_grouping_and_shrinkage_experiments')
Head = _grouping.Head
train_one = _grouping.train_one
compute_metrics = _grouping.compute_metrics


# ============================================================
# 1. JointModel + demographic dataloaders
# ============================================================
class JointModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.feat = None
        # Hook the 768-dim feature (output of VIT se_block)
        self.hook_handle = self.base.VIT.se_block.register_forward_hook(self._hook_fn)
        # Auxiliary demographic head
        self.demog_trunk = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.race_head = nn.Linear(256, N_RACE)
        self.gender_head = nn.Linear(256, N_GENDER)
        self.age_head = nn.Linear(256, N_AGE)

    def _hook_fn(self, module, inp, out):
        self.feat = out

    def forward(self, x):
        emo_logits = self.base(x)
        h = self.demog_trunk(self.feat)
        return emo_logits, self.race_head(h), self.gender_head(h), self.age_head(h)


def build_demog_loaders(demo_map, batch_size=48, num_workers=2):
    train_demog_ds = RafDbDemogDataset(os.path.join(CSV_DIR, 'train_labels.csv'), RAFDB_IMG_DIR, demo_map, train_transform)
    val_demog_ds = RafDbDemogDataset(os.path.join(CSV_DIR, 'valid_labels.csv'), RAFDB_IMG_DIR, demo_map, eval_transform)
    train_demog_loader = DataLoader(train_demog_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_demog_loader = DataLoader(val_demog_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_demog_ds, val_demog_ds, train_demog_loader, val_demog_loader


# ============================================================
# 2. Lambda sweep training
# ============================================================
def evaluate_joint_model(model, loader):
    model.eval()
    emo_correct, race_correct = 0, 0
    total = 0
    with torch.no_grad():
        for imgs, emos, races, genders, ages in loader:
            imgs, emos, races = imgs.to(device), emos.to(device), races.to(device)
            out_emo, out_race, out_gen, out_age = model(imgs)
            emo_correct += (out_emo.argmax(1) == emos).sum().item()
            race_correct += (out_race.argmax(1) == races).sum().item()
            total += emos.size(0)
    return emo_correct / total, race_correct / total


def run_lambda_sweep(train_demog_loader, val_demog_loader, race_counts, gender_counts, age_counts,
                      lambdas=(0.0, 0.05, 0.2, 0.5, 1.0), epochs_per_lambda=10):
    pyramid_trans_expr2 = load_pyramid_model()
    results = []
    SWEEP_CACHE = os.path.join(POSTER_VAR_CACHE, 'lambda_sweep')
    os.makedirs(SWEEP_CACHE, exist_ok=True)

    wr = (race_counts.sum() / (N_RACE * race_counts.clamp(min=1))).to(device)
    wg = (gender_counts.sum() / (N_GENDER * gender_counts.clamp(min=1))).to(device)
    wa = (age_counts.sum() / (N_AGE * age_counts.clamp(min=1))).to(device)

    for lmbda in lambdas:
        print(f"\n{'=' * 40}\nStarting sweep for lambda = {lmbda}\n{'=' * 40}")
        base = pyramid_trans_expr2(img_size=224, num_classes=7, vae=True).to(device)
        base.load_state_dict(torch.load(EMA_CHECKPOINT_PATH, map_location=device))

        # Freeze backbone (ir_back) to prevent degradation of early signals
        for param in base.ir_back.parameters():
            param.requires_grad = False

        joint_model = JointModel(base).to(device)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, joint_model.parameters()), lr=1e-4)

        start_time = time.time()
        best_val_emo = 0
        val_emo, val_race = 0.0, 0.0
        for epoch in range(epochs_per_lambda):
            joint_model.train()
            train_loss, tr_emo_corr, tr_race_corr, tr_total = 0, 0, 0, 0
            for imgs, emos, races, genders, ages in train_demog_loader:
                imgs = imgs.to(device)
                emos, races, genders, ages = emos.to(device), races.to(device), genders.to(device), ages.to(device)
                opt.zero_grad()
                out_emo, out_race, out_gen, out_age = joint_model(imgs)
                loss_emo = F.cross_entropy(out_emo, emos)
                loss_race = F.cross_entropy(out_race, races, weight=wr)
                loss_gen = F.cross_entropy(out_gen, genders, weight=wg)
                loss_age = F.cross_entropy(out_age, ages, weight=wa)
                demog_loss = loss_race + loss_gen + loss_age
                loss = loss_emo + lmbda * demog_loss
                loss.backward()
                opt.step()

                train_loss += loss.item()
                tr_emo_corr += (out_emo.argmax(1) == emos).sum().item()
                tr_race_corr += (out_race.argmax(1) == races).sum().item()
                tr_total += emos.size(0)

            val_emo, val_race = evaluate_joint_model(joint_model, val_demog_loader)
            print(f"  Epoch {epoch + 1}/{epochs_per_lambda} | Loss: {train_loss / max(1, len(train_demog_loader)):.3f} | "
                  f"Tr Emo: {tr_emo_corr / tr_total:.3f}, Tr Race: {tr_race_corr / tr_total:.3f} | "
                  f"Val Emo: {val_emo:.3f}, Val Race: {val_race:.3f}")
            if val_emo > best_val_emo:
                best_val_emo = val_emo
                torch.save(joint_model.state_dict(), os.path.join(SWEEP_CACHE, f'joint_model_lambda_{lmbda}.pth'))

        results.append({
            'lambda': lmbda,
            'val_emo_acc': best_val_emo,
            'val_race_acc': val_race,
            'time_taken': time.time() - start_time
        })

    sweep_results = pd.DataFrame(results)
    sweep_results.to_csv(POSTER_VAR_CACHE + 'lambda_sweep_results.csv', index=False)
    print("Saved lambda sweep results table to Drive.")
    return sweep_results


def list_lambda_sweep_checkpoints():
    SWEEP_CACHE = os.path.join(POSTER_VAR_CACHE, 'lambda_sweep')
    print("Files already in lambda_sweep on Drive:")
    for f in os.listdir(SWEEP_CACHE):
        full = os.path.join(SWEEP_CACHE, f)
        print(f"  {f}  ({os.path.getsize(full) / 1e6:.1f} MB)")


# ============================================================
# 3. Evaluate the lambda=0.1 checkpoint on test
# ============================================================
def load_joint_01():
    pyramid_trans_expr2 = load_pyramid_model()
    SWEEP_CACHE = os.path.join(POSTER_VAR_CACHE, 'lambda_sweep')
    base_01 = pyramid_trans_expr2(img_size=224, num_classes=7, vae=True).to(device)
    base_01.load_state_dict(torch.load(EMA_CHECKPOINT_PATH, map_location=device))
    for param in base_01.ir_back.parameters():
        param.requires_grad = False
    joint_01 = JointModel(base_01).to(device)
    joint_01.load_state_dict(torch.load(os.path.join(SWEEP_CACHE, 'joint_model_lambda_0.1.pth'), map_location=device))
    joint_01.eval()
    return joint_01


def evaluate_lambda01_on_test(joint_01, demo_map, batch_size=48, num_workers=2):
    test_demog_ds = RafDbDemogDataset(os.path.join(CSV_DIR, 'test_labels.csv'), RAFDB_IMG_DIR, demo_map, eval_transform)
    test_demog_loader = DataLoader(test_demog_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_emo_acc, test_race_acc = evaluate_joint_model(joint_01, test_demog_loader)
    print(f"Lambda 0.1 on TEST set: Emotion accuracy {test_emo_acc:.4f}, Race accuracy {test_race_acc:.4f}")
    print("\nFor comparison:")
    print("  Original POSTER-Var (no demog objective) test emotion accuracy: 0.8970")
    print("  Original POSTER-Var race decodability (from decodability experiment): ~0.024 (Black), ~0.001 (Asian)")
    return test_emo_acc, test_race_acc, test_demog_loader


@torch.no_grad()
def evaluate_joint_model_per_group(model, loader):
    model.eval()
    all_race_true, all_race_pred = [], []
    emo_correct, total = 0, 0
    for imgs, emos, races, genders, ages in loader:
        imgs, emos, races = imgs.to(device), emos.to(device), races.to(device)
        out_emo, out_race, out_gen, out_age = model(imgs)
        emo_correct += (out_emo.argmax(1) == emos).sum().item()
        total += emos.size(0)
        all_race_true.append(races.cpu())
        all_race_pred.append(out_race.argmax(1).cpu())
    all_race_true = torch.cat(all_race_true)
    all_race_pred = torch.cat(all_race_pred)
    from common_utils import per_group_accuracy
    per_group = per_group_accuracy(all_race_pred, all_race_true, N_RACE)
    return emo_correct / total, per_group


def report_lambda01_per_group(joint_01, test_demog_loader):
    test_emo_acc, race_per_group = evaluate_joint_model_per_group(joint_01, test_demog_loader)
    print("Lambda 0.1 on TEST set:")
    print(f"  Emotion accuracy: {test_emo_acc:.4f}")
    print(f"  Race per-group (White, Black, Asian): {[round(x, 3) for x in race_per_group]}")
    print("\nCompare to original POSTER-Var (no demog objective), from the 10-seed decodability sweep:")
    print("  Race per-group: White ~0.996, Black ~0.024, Asian ~0.001")
    return test_emo_acc, race_per_group


# ============================================================
# 4. Re-extract features from the lambda=0.1 model + re-attach demographics
# ============================================================
def extract_lambda01_features(joint_01, csv_name):
    captured_01 = {}

    def hook_fn_01(module, inp, out):
        captured_01['feat'] = out.detach()

    hook_handle_01 = joint_01.base.VIT.se_block.register_forward_hook(hook_fn_01)

    ds = RafDbDataset(os.path.join(CSV_DIR, csv_name), RAFDB_IMG_DIR, eval_transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    feats, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            _ = joint_01.base(imgs)  # forward through base triggers the se_block hook
            feats.append(captured_01['feat'].cpu())
            labels_all.append(labels)
    hook_handle_01.remove()
    return torch.cat(feats, dim=0), torch.cat(labels_all, dim=0)


def attach_demographics_lambda01(feats, emo, csv_name, demo_map):
    order_df = pd.read_csv(os.path.join(CSV_DIR, csv_name))
    races, genders, ages = [], [], []
    for fname in order_df['filename']:
        d = demo_map[fname]
        races.append(d['race']); genders.append(d['gender']); ages.append(d['age'])
    return {
        'features': feats.to(device), 'emotion': emo.to(device),
        'race': torch.tensor(races).to(device), 'gender': torch.tensor(genders).to(device),
        'age': torch.tensor(ages).to(device)
    }


def build_and_cache_lambda01_features(joint_01, demo_map):
    print("Extracting lambda 0.1 features for train, val, test...")
    lam01_train_feat, lam01_train_emo = extract_lambda01_features(joint_01, 'train_labels.csv')
    lam01_val_feat, lam01_val_emo = extract_lambda01_features(joint_01, 'valid_labels.csv')
    lam01_test_feat, lam01_test_emo = extract_lambda01_features(joint_01, 'test_labels.csv')
    print(f"Train: {lam01_train_feat.shape}, Val: {lam01_val_feat.shape}, Test: {lam01_test_feat.shape}")

    train_d_lam01 = attach_demographics_lambda01(lam01_train_feat, lam01_train_emo, 'train_labels.csv', demo_map)
    val_d_lam01 = attach_demographics_lambda01(lam01_val_feat, lam01_val_emo, 'valid_labels.csv', demo_map)
    test_d_lam01 = attach_demographics_lambda01(lam01_test_feat, lam01_test_emo, 'test_labels.csv', demo_map)

    torch.save({'features': lam01_train_feat, 'labels': lam01_train_emo}, FEATURE_CACHE + 'posterv2_lambda01_train.pt')
    torch.save({'features': lam01_val_feat, 'labels': lam01_val_emo}, FEATURE_CACHE + 'posterv2_lambda01_val.pt')
    torch.save({'features': lam01_test_feat, 'labels': lam01_test_emo}, FEATURE_CACHE + 'posterv2_lambda01_test.pt')
    print("Cached to Drive.")

    return train_d_lam01, val_d_lam01, test_d_lam01


# ============================================================
# 5. Demographic-conditioning comparison on the recovered features
# ============================================================
def run_lambda01_conditioning_sweep(train_d_lam01, val_d_lam01, test_d_lam01, n_seeds=10):
    race_counts = torch.bincount(train_d_lam01['race'], minlength=N_RACE).float()
    gender_counts = torch.bincount(train_d_lam01['gender'], minlength=N_GENDER).float()
    age_counts = torch.bincount(train_d_lam01['age'], minlength=N_AGE).float()
    emo_counts = torch.bincount(train_d_lam01['emotion'], minlength=7).float()
    class_w = (emo_counts.sum() / (7 * emo_counts)).to(device)

    records_lam01 = []
    for mode in ['none', 'hard_fine']:
        for seed in range(n_seeds):
            model = train_one(mode, seed, train_d_lam01, val_d_lam01, race_counts, gender_counts, age_counts, class_w)
            m = compute_metrics(model, test_d_lam01)
            m.update({'mode': mode, 'seed': seed, 'features': 'lambda_0.1'})
            records_lam01.append(m)
        print(f"{mode} done")

    lam01_df = pd.DataFrame(records_lam01)
    lam01_df.to_csv(POSTER_VAR_CACHE + 'lambda01_conditioning_10seed.csv', index=False)

    print("\nLambda 0.1 features, test-split means:")
    print(lam01_df.groupby('mode')[['overall', 'macro_race_nr', 'worst_race_nr', 'gap_race_nr']].mean())
    return lam01_df


# ============================================================
# 6. Rigorous statistical battery on hard_fine vs none (lambda=0.1 features)
# ============================================================
def rigorous_paired_test(lam01_df, metric, mode_a, mode_b, n_boot=10000, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    a = lam01_df[lam01_df['mode'] == mode_a].sort_values('seed')[metric].reset_index(drop=True)
    b = lam01_df[lam01_df['mode'] == mode_b].sort_values('seed')[metric].reset_index(drop=True)
    diff = a - b
    n = len(diff)

    # 1. Normality check on the paired differences
    shapiro_stat, shapiro_p = stats.shapiro(diff)
    # 2. Paired t-test (parametric)
    t_stat, t_p = stats.ttest_rel(a, b)
    # 3. Wilcoxon signed-rank (nonparametric)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    # 4. Permutation test on the paired differences (sign-flip)
    perm_means = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        perm_means[i] = (diff * signs).mean()
    perm_p = (np.abs(perm_means) >= np.abs(diff.mean())).mean()
    # 5. Bootstrap confidence interval on the mean difference (percentile method)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = diff[idx].mean()
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    # 6. Paired Cohen's d_z
    dz = diff.mean() / diff.std(ddof=1)
    boot_dz = boot_means / diff.std(ddof=1)  # approximate, holding sd fixed
    dz_ci_low, dz_ci_high = np.percentile(boot_dz, [2.5, 97.5])

    return {
        'metric': metric, 'n': n,
        'mean_a': a.mean(), 'mean_b': b.mean(), 'mean_diff': diff.mean(),
        'ci95_low': ci_low, 'ci95_high': ci_high,
        'shapiro_p': shapiro_p, 'normal_ok': shapiro_p > 0.05,
        't_p': t_p, 'wilcoxon_p': w_p, 'permutation_p': perm_p,
        'cohen_dz': dz, 'dz_ci_low': dz_ci_low, 'dz_ci_high': dz_ci_high,
    }


def holm_bonferroni(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m)
    running_max = 0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    return adj


def run_lambda01_rigorous_stats():
    lam01_df = pd.read_csv(POSTER_VAR_CACHE + 'lambda01_conditioning_10seed.csv')
    metrics = ['overall', 'macro_race_nr', 'worst_race_nr', 'gap_race_nr']

    print("=== hard_fine vs none, lambda 0.1 (recovered) features, rigorous battery ===")
    rows = [rigorous_paired_test(lam01_df, m, 'hard_fine', 'none') for m in metrics]
    for pkey in ['t_p', 'wilcoxon_p', 'permutation_p']:
        pv = [r[pkey] for r in rows]
        adj = holm_bonferroni(pv)
        for r, a in zip(rows, adj):
            r[f'holm_{pkey}'] = a

    res = pd.DataFrame(rows)
    display_cols = ['metric', 'mean_diff', 'ci95_low', 'ci95_high', 'shapiro_p', 'normal_ok',
                     't_p', 'holm_t_p', 'wilcoxon_p', 'holm_wilcoxon_p',
                     'permutation_p', 'holm_permutation_p', 'cohen_dz', 'dz_ci_low', 'dz_ci_high']
    print(res[display_cols].round(4).to_string(index=False))
    res.to_csv(POSTER_VAR_CACHE + 'lambda01_rigorous_stats.csv', index=False)
    print(f"\nSaved to {POSTER_VAR_CACHE}lambda01_rigorous_stats.csv")
    return res


if __name__ == '__main__':
    demo_df = pd.read_csv(POSTER_VAR_CACHE + 'rafdb_demographics.csv')
    demo_map = demo_df.set_index('filename')[['race', 'gender', 'age']].to_dict('index')

    train_demog_ds, val_demog_ds, train_demog_loader, val_demog_loader = build_demog_loaders(demo_map)

    train_full = torch.load(FEATURE_CACHE + 'posterv2_ema_train_full.pt')
    race_counts = torch.bincount(train_full['race'], minlength=N_RACE).float()
    gender_counts = torch.bincount(train_full['gender'], minlength=N_GENDER).float()
    age_counts = torch.bincount(train_full['age'], minlength=N_AGE).float()

    run_lambda_sweep(train_demog_loader, val_demog_loader, race_counts, gender_counts, age_counts,
                      lambdas=(0.0, 0.1, 0.4), epochs_per_lambda=10)

    joint_01 = load_joint_01()
    test_emo_acc, test_race_acc, test_demog_loader = evaluate_lambda01_on_test(joint_01, demo_map)
    report_lambda01_per_group(joint_01, test_demog_loader)

    train_d_lam01, val_d_lam01, test_d_lam01 = build_and_cache_lambda01_features(joint_01, demo_map)
    run_lambda01_conditioning_sweep(train_d_lam01, val_d_lam01, test_d_lam01)
    run_lambda01_rigorous_stats()
