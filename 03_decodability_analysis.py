"""
03_decodability_analysis.py

  1. decode_race() — 3-point decodability comparison (pristine IR50, IR50
     stream inside POSTER-Var, POSTER-Var final feature), 5 seeds each.
  2. Weighted demographic classifier trained on cached POSTER-Var features,
     per-group accuracy for race/gender/age.
  3. Unweighted demographic classifier + confusion matrices (majority-class
     baseline check).
  4. Pristine-IR50 vs POSTER-Var decodability, 10-seed paired comparison
     with a t-test and Cohen's d on Black/Asian race recall.
  5. Simple linear probe of emotion accuracy directly on pristine IR50
     features, for reference against POSTER-Var's accuracy.

Requires the cached feature files produced by 02_feature_extraction.py and
01_setup_and_prepare_data.py (the *_full.pt files with demographics attached).
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from common_utils import (
    FEATURE_CACHE, POSTER_VAR_CACHE, N_RACE, N_GENDER, N_AGE, device,
    to_device, DemogClf, DemogClassifier, per_group_accuracy,
)

# ------------------------------------------------------------------
# Load demographic-tagged cached features (produced upstream)
# ------------------------------------------------------------------
def load_full_features():
    train_full = torch.load(FEATURE_CACHE + 'posterv2_ema_train_full.pt')
    val_full = torch.load(FEATURE_CACHE + 'posterv2_ema_val_full.pt')
    test_full = torch.load(FEATURE_CACHE + 'posterv2_ema_test_full.pt')
    train_d, val_d, test_d = to_device(train_full), to_device(val_full), to_device(test_full)

    race_counts = torch.bincount(train_d['race'], minlength=N_RACE).float()
    gender_counts = torch.bincount(train_d['gender'], minlength=N_GENDER).float()
    age_counts = torch.bincount(train_d['age'], minlength=N_AGE).float()
    emo_counts = torch.bincount(train_d['emotion'], minlength=7).float()
    class_w = (emo_counts.sum() / (7 * emo_counts)).to(device)

    print("Feature tensors loaded:")
    print(f"  train {train_d['features'].shape}, val {val_d['features'].shape}, test {test_d['features'].shape}")
    return train_d, val_d, test_d, race_counts, gender_counts, age_counts, class_w


# ============================================================
# 1. Three-point race decodability comparison along the pipeline
# ============================================================
def decode_race(Xtr, Xte, tag, train_d, test_d, seeds=5):
    accs_black, accs_asian, accs_overall = [], [], []
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        clf = DemogClf(Xtr.shape[1]).to(device)
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=0.05)
        n = Xtr.shape[0]; bs = 256
        for _ in range(40):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; opt.zero_grad()
                pr, pg, pa = clf(Xtr[idx])
                (F.cross_entropy(pr, train_d['race'][idx])
                 + F.cross_entropy(pg, train_d['gender'][idx])
                 + F.cross_entropy(pa, train_d['age'][idx])).backward()
                opt.step()
        clf.eval()
        with torch.no_grad():
            pr = clf(Xte)[0].argmax(1)
        rg = per_group_accuracy(pr, test_d['race'], N_RACE)
        accs_black.append(rg[1]); accs_asian.append(rg[2])
        accs_overall.append((pr == test_d['race']).float().mean().item())
    print(f"{tag}:  Black {np.mean(accs_black):.3f}+/-{np.std(accs_black):.3f}   "
          f"Asian {np.mean(accs_asian):.3f}+/-{np.std(accs_asian):.3f}   "
          f"overall {np.mean(accs_overall):.3f}")
    return accs_black, accs_asian, accs_overall


def run_pipeline_decodability(ir_train, ir_test, irstream_train, irstream_test, train_d, test_d):
    print("\nRace decodability along the pipeline:")
    decode_race(ir_train, ir_test, "1. Pristine IR50 (pre-expression)   ", train_d, test_d)
    decode_race(irstream_train, irstream_test, "2. IR50 stream inside POSTER-Var   ", train_d, test_d)
    decode_race(train_d['features'], test_d['features'], "3. POSTER-Var final feature (768-d)", train_d, test_d)


# ============================================================
# 2. Weighted demographic classifier (inverse-frequency class weights)
# ============================================================
def train_demog_classifier(train_d, race_counts, gender_counts, age_counts, seed=0, epochs=40):
    torch.manual_seed(seed); np.random.seed(seed)
    clf = DemogClassifier().to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=0.05)

    def w(counts):
        return (counts.sum() / (len(counts) * counts.clamp(min=1))).to(device)

    wr, wg, wa = w(race_counts), w(gender_counts), w(age_counts)
    n = train_d['features'].shape[0]; bs = 256
    for _ in range(epochs):
        clf.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            pr, pg, pa = clf(train_d['features'][idx])
            loss = (F.cross_entropy(pr, train_d['race'][idx], weight=wr)
                    + F.cross_entropy(pg, train_d['gender'][idx], weight=wg)
                    + F.cross_entropy(pa, train_d['age'][idx], weight=wa))
            loss.backward(); opt.step()
    return clf


@torch.no_grad()
def predict_demographics(clf, d):
    clf.eval()
    pr, pg, pa = clf(d['features'])
    return pr.argmax(1), pg.argmax(1), pa.argmax(1)


def run_weighted_classifier_report(train_d, val_d, test_d, race_counts, gender_counts, age_counts):
    clf = train_demog_classifier(train_d, race_counts, gender_counts, age_counts, seed=0)
    for split_name, d in [('train', train_d), ('val', val_d), ('test', test_d)]:
        pr, pg, pa = predict_demographics(clf, d)
        ra = per_group_accuracy(pr, d['race'], N_RACE)
        ga = per_group_accuracy(pg, d['gender'], N_GENDER)
        aa = per_group_accuracy(pa, d['age'], N_AGE)
        print(f"\n{split_name} demographic classifier accuracy:")
        print(f"  race  per-group: {[round(x, 3) for x in ra]}  overall {(pr == d['race']).float().mean():.3f}")
        print(f"  gender per-group: {[round(x, 3) for x in ga]}  overall {(pg == d['gender']).float().mean():.3f}")
        print(f"  age   per-group: {[round(x, 3) for x in aa]}  overall {(pa == d['age']).float().mean():.3f}")
    return clf


# ============================================================
# 3. Unweighted demographic classifier + confusion matrices
# ============================================================
def train_demog_unweighted(train_d, seed=0, epochs=40):
    torch.manual_seed(seed); np.random.seed(seed)
    clf = DemogClassifier().to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=0.05)
    n = train_d['features'].shape[0]; bs = 256
    for _ in range(epochs):
        clf.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            pr, pg, pa = clf(train_d['features'][idx])
            loss = (F.cross_entropy(pr, train_d['race'][idx])
                    + F.cross_entropy(pg, train_d['gender'][idx])
                    + F.cross_entropy(pa, train_d['age'][idx]))
            loss.backward(); opt.step()
    return clf


def run_unweighted_classifier_report(train_d, test_d):
    clf_uw = train_demog_unweighted(train_d, seed=0)
    d = test_d
    pr, pg, pa = predict_demographics(clf_uw, d)
    print("\ntest UNWEIGHTED demographic classifier:")
    print(f"  race   overall {(pr == d['race']).float().mean():.3f}")
    print(f"  gender overall {(pg == d['gender']).float().mean():.3f}")
    print(f"  age    overall {(pa == d['age']).float().mean():.3f}")

    print("\n  race confusion (rows true, cols pred):")
    cm = torch.zeros(N_RACE, N_RACE, dtype=torch.int)
    for t, p in zip(d['race'].cpu(), pr.cpu()):
        cm[t, p] += 1
    print(cm.numpy())

    print("Per-group recall, unweighted classifier (test):")
    print("  race  :", [round(x, 3) for x in per_group_accuracy(pr, d['race'], N_RACE)])
    print("  gender:", [round(x, 3) for x in per_group_accuracy(pg, d['gender'], N_GENDER)])
    print("  age   :", [round(x, 3) for x in per_group_accuracy(pa, d['age'], N_AGE)])

    for name, pred, true, n in [('gender', pg, d['gender'], N_GENDER), ('age', pa, d['age'], N_AGE)]:
        cm = torch.zeros(n, n, dtype=torch.int)
        for t, p in zip(true.cpu(), pred.cpu()):
            cm[t, p] += 1
        print(f"\n{name} confusion (rows true, cols pred):")
        print(cm.numpy())
    return clf_uw


# ============================================================
# 4. Pristine IR50 vs POSTER-Var: paired decodability comparison
# ============================================================
def train_and_eval_demog(Xtr, Xte, tag, train_d, test_d, seed=0, epochs=40):
    torch.manual_seed(seed); np.random.seed(seed)
    clf = DemogClf(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=0.05)
    n = Xtr.shape[0]; bs = 256
    for _ in range(epochs):
        clf.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            pr, pg, pa = clf(Xtr[idx])
            loss = (F.cross_entropy(pr, train_d['race'][idx])
                    + F.cross_entropy(pg, train_d['gender'][idx])
                    + F.cross_entropy(pa, train_d['age'][idx]))
            loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        pr, pg, pa = clf(Xte)
        pr, pg, pa = pr.argmax(1), pg.argmax(1), pa.argmax(1)
    r = [round(x, 3) for x in per_group_accuracy(pr, test_d['race'], N_RACE)]
    g = [round(x, 3) for x in per_group_accuracy(pg, test_d['gender'], N_GENDER)]
    a = [round(x, 3) for x in per_group_accuracy(pa, test_d['age'], N_AGE)]
    print(f"\n{tag}:")
    print(f"  race   per-group: {r}  overall {(pr == test_d['race']).float().mean():.3f}")
    print(f"  gender per-group: {g}  overall {(pg == test_d['gender']).float().mean():.3f}")
    print(f"  age    per-group: {a}  overall {(pa == test_d['age']).float().mean():.3f}")
    return clf


def collect_seed_results(Xtr, Xte, tag, seed, train_d, test_d, epochs=40):
    torch.manual_seed(seed); np.random.seed(seed)
    clf = DemogClf(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=0.05)
    n = Xtr.shape[0]; bs = 256
    for _ in range(epochs):
        clf.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            pr, pg, pa = clf(Xtr[idx])
            loss = (F.cross_entropy(pr, train_d['race'][idx])
                    + F.cross_entropy(pg, train_d['gender'][idx])
                    + F.cross_entropy(pa, train_d['age'][idx]))
            loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        pr, pg, pa = clf(Xte)
        pr, pg, pa = pr.argmax(1), pg.argmax(1), pa.argmax(1)
    r = per_group_accuracy(pr, test_d['race'], N_RACE)
    g = per_group_accuracy(pg, test_d['gender'], N_GENDER)
    a = per_group_accuracy(pa, test_d['age'], N_AGE)
    row = {'tag': tag, 'seed': seed,
           'race_overall': (pr == test_d['race']).float().mean().item(),
           'gender_overall': (pg == test_d['gender']).float().mean().item(),
           'age_overall': (pa == test_d['age']).float().mean().item()}
    for i, v in enumerate(r): row[f'race{i}'] = v
    for i, v in enumerate(g): row[f'gender{i}'] = v
    for i, v in enumerate(a): row[f'age{i}'] = v
    return row


def run_demog_decodability_seed_sweep(ir_train, ir_test, train_d, test_d, n_seeds=10):
    records = []
    for seed in range(n_seeds):
        records.append(collect_seed_results(ir_train, ir_test, 'pristine_ir50', seed, train_d, test_d))
        records.append(collect_seed_results(train_d['features'], test_d['features'], 'poster_var', seed, train_d, test_d))
        print(f"seed {seed} done")

    demog_seed_df = pd.DataFrame(records)
    demog_seed_df.to_csv(POSTER_VAR_CACHE + 'demog_decodability_10seed.csv', index=False)

    print("\nMean +/- std across seeds:")
    cols = ['race_overall', 'race0', 'race1', 'race2', 'age_overall', 'age0', 'age1',
            'gender_overall', 'gender0', 'gender1', 'gender2']
    # NOTE: preserved from source; some age columns beyond age1 exist too but
    # were omitted from the printed summary in the original notebook.
    cols = [c for c in cols if c in demog_seed_df.columns]
    summary = demog_seed_df.groupby('tag')[cols].agg(['mean', 'std'])
    print(summary.round(3).to_string())

    for metric in ['race1', 'race2', 'race_overall']:
        a = demog_seed_df[demog_seed_df.tag == 'pristine_ir50'].sort_values('seed')[metric]
        b = demog_seed_df[demog_seed_df.tag == 'poster_var'].sort_values('seed')[metric]
        t, p = stats.ttest_rel(a, b)
        d = (a - b).mean() / (a - b).std(ddof=1)
        print(f"\n{metric}: pristine {a.mean():.3f}, poster {b.mean():.3f}, "
              f"t-test p={p:.2e}, Cohen's d={d:.2f}")
    return demog_seed_df


# ============================================================
# 5. Linear probe: emotion accuracy directly on pristine IR50 features
# ============================================================
def ir50_linear_probe_emotion_accuracy(ir_train, ir_test, train_d, test_d):
    probe_ir = nn.Linear(ir_train.shape[1], 7).to(device)
    opt = torch.optim.AdamW(probe_ir.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    n = ir_train.shape[0]
    for epoch in range(30):
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = crit(probe_ir(ir_train[idx]), train_d['emotion'][idx])
            loss.backward()
            opt.step()
    probe_ir.eval()
    with torch.no_grad():
        test_acc = (probe_ir(ir_test).argmax(1) == test_d['emotion']).float().mean()
    print(f"IR50 linear probe emotion accuracy: {test_acc:.4f}")
    print("(for reference, POSTER-Var linear probe was 0.9130)")
    return test_acc


if __name__ == '__main__':
    train_d, val_d, test_d, race_counts, gender_counts, age_counts, class_w = load_full_features()
    run_weighted_classifier_report(train_d, val_d, test_d, race_counts, gender_counts, age_counts)
    run_unweighted_classifier_report(train_d, test_d)
    # ir_train / ir_test / irstream_train / irstream_test come from
    # 02_feature_extraction.py — load or re-extract them before calling:
    # run_pipeline_decodability(ir_train, ir_test, irstream_train, irstream_test, train_d, test_d)
    # run_demog_decodability_seed_sweep(ir_train, ir_test, train_d, test_d)
    # ir50_linear_probe_emotion_accuracy(ir_train, ir_test, train_d, test_d)
