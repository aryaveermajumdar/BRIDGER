"""
04_grouping_and_shrinkage_experiments.py

Consolidates the demographic-conditioning "grouping" experiments:

  1. Partition-search helpers (all_partitions_of_set, contiguous_partitions)
     and the brute-force race/age partition search via ConditionedProbe.
     NOTE: in the original notebook the actual 80-candidate search loop was
     left commented out (superseded by the staged-training approach below).
     It is preserved here, still inactive, exactly as authored.
  2. ShrinkageEmbedding / ShrinkageConditionedHead + the 3-stage training
     protocol (train_staged) — the adaptive per-group shrinkage approach.
  3. Head (mode: none | hard_fine | hard_best | shrinkage) — the head used
     for the main 20-seed comparison sweep, plus compute_metrics/train_one.
  4. The full sweep across all four modes -> grouping_results_20seed.csv.
  5. Statistical comparison of shrinkage vs none / shrinkage vs hard_fine
     (paired t-test, Wilcoxon, Holm correction).
  6. Sparsity sweep helpers (cap_marginal / cap_intersectional / the sweep
     loop itself — also left inactive in the source notebook) plus the
     analysis of the resulting sparsity_sweep_10seed.csv.

Requires the *_full.pt feature caches from earlier scripts.
"""

import time
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from common_utils import (
    FEATURE_CACHE, POSTER_VAR_CACHE, NEUTRAL_CLASS, N_RACE, N_GENDER, N_AGE,
    GENDER_GROUPS, device, to_device, per_group_accuracy,
)


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
    return train_d, val_d, test_d, race_counts, gender_counts, age_counts, class_w


# ============================================================
# 1. Partition search (INACTIVE in the source notebook — preserved as-is)
# ============================================================
#
# import torch
# import torch.nn as nn
# from itertools import combinations
# import numpy as np
# from collections import defaultdict
#
# FEATURE_CACHE = POSTER_VAR_CACHE + 'features/'
# train_full = torch.load(FEATURE_CACHE + 'posterv2_ema_train_full.pt')
# val_full   = torch.load(FEATURE_CACHE + 'posterv2_ema_val_full.pt')
# NEUTRAL_CLASS = 6  # RAF-DB 0-indexed: 6 = Neutral
#
# # ---------- partition generators ----------
# def all_partitions_of_set(elements):
#     # every way to partition a small set (used for race, 3 elements)
#     elements = list(elements)
#     if len(elements) == 1:
#         yield [elements]
#         return
#     first = elements[0]
#     for smaller in all_partitions_of_set(elements[1:]):
#         for i, subset in enumerate(smaller):
#             yield smaller[:i] + [[first] + subset] + smaller[i+1:]
#         yield [[first]] + smaller
#
# def contiguous_partitions(n):
#     # every way to split ordered bins 0..n-1 into contiguous groups
#     # determined by which of the n-1 gaps are "cut"
#     results = []
#     gaps = n - 1
#     for mask in range(2 ** gaps):
#         groups, cur = [], [0]
#         for g in range(gaps):
#             if mask & (1 << g):
#                 groups.append(cur); cur = [g+1]
#             else:
#                 cur.append(g+1)
#         groups.append(cur)
#         results.append(groups)
#     return results
#
# def partition_to_map(partition):
#     # convert a list of groups into a dict {original_label: group_id}
#     m = {}
#     for gid, group in enumerate(partition):
#         for label in group:
#             m[label] = gid
#     return m
#
# race_partitions = list(all_partitions_of_set([0, 1, 2]))
# age_partitions  = contiguous_partitions(5)
# print(f"Race partitions: {len(race_partitions)}, Age partitions: {len(age_partitions)}")
# print(f"Total joint candidates: {len(race_partitions) * len(age_partitions)}")


def all_partitions_of_set(elements):
    """Active helper — every way to partition a small set (used for race, 3 elements)."""
    elements = list(elements)
    if len(elements) == 1:
        yield [elements]
        return
    first = elements[0]
    for smaller in all_partitions_of_set(elements[1:]):
        for i, subset in enumerate(smaller):
            yield smaller[:i] + [[first] + subset] + smaller[i + 1:]
        yield [[first]] + smaller


def contiguous_partitions(n):
    """Every way to split ordered bins 0..n-1 into contiguous groups,
    determined by which of the n-1 gaps are 'cut'."""
    results = []
    gaps = n - 1
    for mask in range(2 ** gaps):
        groups, cur = [], [0]
        for g in range(gaps):
            if mask & (1 << g):
                groups.append(cur); cur = [g + 1]
            else:
                cur.append(g + 1)
        groups.append(cur)
        results.append(groups)
    return results


def partition_to_map(partition):
    m = {}
    for gid, group in enumerate(partition):
        for label in group:
            m[label] = gid
    return m


class ConditionedProbe(nn.Module):
    def __init__(self, n_race, n_gender, n_age, emb_dim=16):
        super().__init__()
        self.race_emb = nn.Embedding(n_race, emb_dim)
        self.gender_emb = nn.Embedding(n_gender, emb_dim)
        self.age_emb = nn.Embedding(n_age, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(768 + 3 * emb_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 7)
        )

    def forward(self, f, r, g, a):
        z = torch.cat([f, self.race_emb(r), self.gender_emb(g), self.age_emb(a)], dim=1)
        return self.mlp(z)


def remap(labels, pmap):
    return torch.tensor([pmap[int(x)] for x in labels], device=device)


def neutral_recall_by_group(preds, emo, group_ids, n_groups):
    recalls = []
    for gid in range(n_groups):
        mask = (group_ids == gid) & (emo == NEUTRAL_CLASS)
        if mask.sum() == 0:
            recalls.append(None); continue
        correct = ((preds == NEUTRAL_CLASS) & mask).sum().item()
        recalls.append(correct / mask.sum().item())
    return recalls


def evaluate_candidate(race_part, age_part, train_d, val_d, epochs=25, seed=0):
    torch.manual_seed(seed)
    rmap, amap = partition_to_map(race_part), partition_to_map(age_part)
    n_race, n_age = len(race_part), len(age_part)
    tr_r = remap(train_d['race'], rmap); tr_g = train_d['gender'].to(device); tr_a = remap(train_d['age'], amap)
    va_r = remap(val_d['race'], rmap); va_g = val_d['gender'].to(device); va_a = remap(val_d['age'], amap)

    race_group_sizes = [(tr_r == gid).sum().item() for gid in range(n_race)]
    min_race_group = min(race_group_sizes)

    probe = ConditionedProbe(n_race, GENDER_GROUPS, n_age).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.05)
    crit = nn.CrossEntropyLoss()
    n = train_d['features'].shape[0]
    bs = 256
    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = probe(train_d['features'][idx], tr_r[idx], tr_g[idx], tr_a[idx])
            crit(out, train_d['emotion'][idx]).backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(val_d['features'], va_r, va_g, va_a).argmax(1)
        overall_acc = (preds == val_d['emotion']).float().mean().item()
        race_neutral = neutral_recall_by_group(preds, val_d['emotion'], va_r, n_race)
        valid = [x for x in race_neutral if x is not None]
        worst_race_neutral = min(valid) if valid else 0.0
        macro_race_neutral = sum(valid) / len(valid) if valid else 0.0

    return {
        'race_part': race_part, 'age_part': age_part,
        'n_race': n_race, 'n_age': n_age,
        'overall_acc': overall_acc,
        'worst_race_neutral': worst_race_neutral,
        'macro_race_neutral': macro_race_neutral,
        'min_race_group': min_race_group,
    }


def run_partition_search(train_d, val_d):
    """The full 80-candidate search (active version of the cell above)."""
    race_partitions = list(all_partitions_of_set([0, 1, 2]))
    age_partitions = contiguous_partitions(5)
    print(f"Race partitions: {len(race_partitions)}, Age partitions: {len(age_partitions)}")
    print(f"Total joint candidates: {len(race_partitions) * len(age_partitions)}")

    results = []
    start = time.time()
    for i, rp in enumerate(race_partitions):
        for ap in age_partitions:
            results.append(evaluate_candidate(rp, ap, train_d, val_d))
        print(f"Race partition {i + 1}/{len(race_partitions)} done, {time.time() - start:.0f}s elapsed")
    print(f"\nSearch complete: {len(results)} candidates in {time.time() - start:.0f}s")

    def part_str(p):
        return "|".join("+".join(str(x) for x in g) for g in p)

    rows = []
    for r in results:
        rows.append({
            'race': part_str(r['race_part']),
            'age': part_str(r['age_part']),
            'overall_acc': round(r['overall_acc'], 4),
            'worst_race_neutral': round(r['worst_race_neutral'], 4),
            'macro_race_neutral': round(r['macro_race_neutral'], 4),
            'min_race_grp': r['min_race_group'],
        })
    df = pd.DataFrame(rows).sort_values('worst_race_neutral', ascending=False)
    pd.set_option('display.max_rows', 20)
    print("Top 15 by worst-group Neutral recall:")
    print(df.head(15).to_string(index=False))
    return df, results


# ============================================================
# 2. Shrinkage embeddings + 3-stage training
# ============================================================
class ShrinkageEmbedding(nn.Module):
    """
    Embedding(i) = shared + alpha_i * delta_i,  alpha_i = n_i / (n_i + tau)
    tau is learnable (log-parameterized, always positive).
    Large tau -> heavy pooling toward shared. Small tau -> separate subgroups.
    """
    def __init__(self, n_groups, emb_dim, counts, init_log_tau=6.0):
        super().__init__()
        self.shared = nn.Parameter(torch.randn(emb_dim) * 0.02)
        self.deltas = nn.Embedding(n_groups, emb_dim)
        nn.init.normal_(self.deltas.weight, std=0.02)
        self.log_tau = nn.Parameter(torch.tensor(init_log_tau))
        self.register_buffer('counts', counts)

    def alphas(self):
        tau = torch.exp(self.log_tau)
        return self.counts / (self.counts + tau)

    def forward(self, ids):
        a = self.alphas()[ids].unsqueeze(1)
        return self.shared.unsqueeze(0) + a * self.deltas(ids)

    def age_smoothness(self):
        d = self.deltas.weight
        return ((d[1:] - d[:-1]) ** 2).sum()


class ShrinkageConditionedHead(nn.Module):
    def __init__(self, race_counts, gender_counts, age_counts, emb_dim=32):
        super().__init__()
        self.race_emb = ShrinkageEmbedding(N_RACE, emb_dim, race_counts)
        self.gender_emb = ShrinkageEmbedding(N_GENDER, emb_dim, gender_counts)
        self.age_emb = ShrinkageEmbedding(N_AGE, emb_dim, age_counts)
        in_dim = 768 + 3 * emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 7)
        )
        self.aux_race = nn.Linear(emb_dim, N_RACE)
        self.aux_gender = nn.Linear(emb_dim, N_GENDER)
        self.aux_age = nn.Linear(emb_dim, N_AGE)

    def embed(self, r, g, a):
        return self.race_emb(r), self.gender_emb(g), self.age_emb(a)

    def forward(self, f, r, g, a):
        er, eg, ea = self.embed(r, g, a)
        return self.mlp(torch.cat([f, er, eg, ea], dim=1))

    def aux_losses(self, r, g, a):
        er, eg, ea = self.embed(r, g, a)
        return (F.cross_entropy(self.aux_race(er), r)
                + F.cross_entropy(self.aux_gender(eg), g)
                + F.cross_entropy(self.aux_age(ea), a))


def run_epoch_stage1(model, opt, train_d, bs=256, smooth_lambda=1e-3):
    model.train()
    n = train_d['features'].shape[0]
    perm = torch.randperm(n)
    for i in range(0, n, bs):
        idx = perm[i:i + bs]
        opt.zero_grad()
        loss = model.aux_losses(train_d['race'][idx], train_d['gender'][idx], train_d['age'][idx])
        loss = loss + smooth_lambda * model.age_emb.age_smoothness()
        loss.backward()
        opt.step()


def run_epoch_emotion(model, opt, train_d, class_w, bs=256, smooth_lambda=1e-3):
    model.train()
    n = train_d['features'].shape[0]
    perm = torch.randperm(n)
    for i in range(0, n, bs):
        idx = perm[i:i + bs]
        opt.zero_grad()
        out = model(train_d['features'][idx], train_d['race'][idx],
                    train_d['gender'][idx], train_d['age'][idx])
        loss = F.cross_entropy(out, train_d['emotion'][idx], weight=class_w)
        loss = loss + smooth_lambda * model.age_emb.age_smoothness()
        loss.backward()
        opt.step()


@torch.no_grad()
def full_eval(model, d):
    model.eval()
    preds = model(d['features'], d['race'], d['gender'], d['age']).argmax(1)
    overall = (preds == d['emotion']).float().mean().item()

    def neutral_recall(mask):
        m = mask & (d['emotion'] == NEUTRAL_CLASS)
        if m.sum() == 0:
            return None
        return (((preds == NEUTRAL_CLASS) & m).sum() / m.sum()).item()

    race_nr = [neutral_recall(d['race'] == i) for i in range(N_RACE)]
    age_nr = [neutral_recall(d['age'] == i) for i in range(N_AGE)]
    rv = [x for x in race_nr if x is not None]
    av = [x for x in age_nr if x is not None]
    return {
        'overall': overall,
        'race_neutral': race_nr, 'age_neutral': age_nr,
        'macro_race_neutral': float(np.mean(rv)), 'worst_race_neutral': float(np.min(rv)) if rv else 0.0,
        'macro_age_neutral': float(np.mean(av)), 'worst_age_neutral': float(np.min(av)) if av else 0.0,
    }


def train_staged(train_d, val_d, race_counts, gender_counts, age_counts, class_w,
                  seed=0, s1_epochs=15, s2_epochs=40, s3_epochs=20):
    torch.manual_seed(seed)
    model = ShrinkageConditionedHead(race_counts, gender_counts, age_counts).to(device)
    emb_params = (list(model.race_emb.parameters()) + list(model.gender_emb.parameters())
                  + list(model.age_emb.parameters()))
    aux_params = (list(model.aux_race.parameters()) + list(model.aux_gender.parameters())
                  + list(model.aux_age.parameters()))
    mlp_params = list(model.mlp.parameters())

    # Stage 1: embeddings and aux heads only
    opt1 = torch.optim.AdamW(emb_params + aux_params, lr=1e-3, weight_decay=0.05)
    for _ in range(s1_epochs):
        run_epoch_stage1(model, opt1, train_d)

    # Stage 2: freeze embeddings, train emotion MLP
    for p in emb_params: p.requires_grad_(False)
    opt2 = torch.optim.AdamW(mlp_params, lr=1e-3, weight_decay=0.05)
    for _ in range(s2_epochs):
        run_epoch_emotion(model, opt2, train_d, class_w)

    # Stage 3: unfreeze everything, joint fine-tune at low LR
    for p in emb_params: p.requires_grad_(True)
    opt3 = torch.optim.AdamW(emb_params + mlp_params, lr=1e-4, weight_decay=0.05)
    best_val, best_state = 0, None
    for _ in range(s3_epochs):
        run_epoch_emotion(model, opt3, train_d, class_w)
        v = full_eval(model, val_d)['overall']
        if v > best_val:
            best_val = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def run_train_staged_demo(train_d, val_d, test_d, race_counts, gender_counts, age_counts, class_w):
    model = train_staged(train_d, val_d, race_counts, gender_counts, age_counts, class_w, seed=0)
    print("\nLearned shrinkage state:")
    for name, emb in [('race', model.race_emb), ('gender', model.gender_emb), ('age', model.age_emb)]:
        tau = torch.exp(emb.log_tau).item()
        print(f"  {name}: tau = {tau:.1f}, alphas = {[round(a, 3) for a in emb.alphas().tolist()]}")
    print("\nValidation:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in full_eval(model, val_d).items()})
    print("\nTest:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in full_eval(model, test_d).items()})
    return model


# ============================================================
# 3. Head (mode: none | hard_fine | hard_best | shrinkage) — main sweep
# ============================================================
BEST_RACE_MAP = {0: 0, 1: 1, 2: 2}          # all separate
BEST_AGE_MAP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}  # 0+1 | 2+3+4


class PlainEmbedding(nn.Module):
    def __init__(self, n_groups, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(n_groups, emb_dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, ids):
        return self.emb(ids)

    def age_smoothness(self):
        return torch.tensor(0.0, device=device)


class Head(nn.Module):
    """mode: 'none' | 'hard_fine' | 'hard_best' | 'shrinkage'"""
    def __init__(self, mode, race_counts, gender_counts, age_counts, emb_dim=32):
        super().__init__()
        self.mode = mode
        self.emb_dim = emb_dim
        if mode == 'none':
            in_dim = 768
        else:
            if mode == 'shrinkage':
                self.race_emb = ShrinkageEmbedding(N_RACE, emb_dim, race_counts)
                self.gender_emb = ShrinkageEmbedding(N_GENDER, emb_dim, gender_counts)
                self.age_emb = ShrinkageEmbedding(N_AGE, emb_dim, age_counts)
                self.n_race, self.n_age = N_RACE, N_AGE
            elif mode == 'hard_fine':
                self.race_emb = PlainEmbedding(N_RACE, emb_dim)
                self.gender_emb = PlainEmbedding(N_GENDER, emb_dim)
                self.age_emb = PlainEmbedding(N_AGE, emb_dim)
                self.n_race, self.n_age = N_RACE, N_AGE
            elif mode == 'hard_best':
                nr = len(set(BEST_RACE_MAP.values())); na = len(set(BEST_AGE_MAP.values()))
                self.race_emb = PlainEmbedding(nr, emb_dim)
                self.gender_emb = PlainEmbedding(N_GENDER, emb_dim)
                self.age_emb = PlainEmbedding(na, emb_dim)
                self.n_race, self.n_age = nr, na
            in_dim = 768 + 3 * emb_dim
            self.aux_race = nn.Linear(emb_dim, self.n_race)
            self.aux_gender = nn.Linear(emb_dim, N_GENDER)
            self.aux_age = nn.Linear(emb_dim, self.n_age)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 7)
        )

    def _map(self, r, g, a):
        if self.mode == 'hard_best':
            r = torch.tensor([BEST_RACE_MAP[int(x)] for x in r], device=device)
            a = torch.tensor([BEST_AGE_MAP[int(x)] for x in a], device=device)
        return r, g, a

    def embed(self, r, g, a):
        r, g, a = self._map(r, g, a)
        return self.race_emb(r), self.gender_emb(g), self.age_emb(a)

    def forward(self, f, r, g, a):
        if self.mode == 'none':
            return self.mlp(f)
        er, eg, ea = self.embed(r, g, a)
        return self.mlp(torch.cat([f, er, eg, ea], dim=1))

    def aux_losses(self, r, g, a):
        r2, g2, a2 = self._map(r, g, a)
        er, eg, ea = self.embed(r, g, a)
        return (F.cross_entropy(self.aux_race(er), r2)
                + F.cross_entropy(self.aux_gender(eg), g2)
                + F.cross_entropy(self.aux_age(ea), a2))

    def age_smoothness(self):
        return self.age_emb.age_smoothness() if self.mode != 'none' else torch.tensor(0.0, device=device)


@torch.no_grad()
def compute_metrics(model, d):
    model.eval()
    logits = model(d['features'], d['race'], d['gender'], d['age'])
    preds = logits.argmax(1)
    emo = d['emotion']
    overall = (preds == emo).float().mean().item()

    f1s = []
    for c in range(7):
        tp = ((preds == c) & (emo == c)).sum().item()
        fp = ((preds == c) & (emo != c)).sum().item()
        fn = ((preds != c) & (emo == c)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    macro_f1 = float(np.mean(f1s))

    def neutral_recall(mask):
        m = mask & (emo == NEUTRAL_CLASS)
        if m.sum() == 0:
            return np.nan
        return (((preds == NEUTRAL_CLASS) & m).sum() / m.sum()).item()

    def neutral_selection_rate(mask):
        if mask.sum() == 0:
            return np.nan
        return ((preds == NEUTRAL_CLASS) & mask).float().sum().item() / mask.float().sum().item()

    race_nr = [neutral_recall(d['race'] == i) for i in range(N_RACE)]
    age_nr = [neutral_recall(d['age'] == i) for i in range(N_AGE)]
    gen_nr = [neutral_recall(d['gender'] == i) for i in range(N_GENDER)]
    race_sel = [neutral_selection_rate(d['race'] == i) for i in range(N_RACE)]

    def summ(vals):
        v = [x for x in vals if not np.isnan(x)]
        return float(np.mean(v)), float(min(v)), float(max(v) - min(v))

    macro_race, worst_race, gap_race = summ(race_nr)
    macro_age, worst_age, gap_age = summ(age_nr)
    macro_gen, worst_gen, gap_gen = summ(gen_nr)
    dp_race = float(np.nanmax(race_sel) - np.nanmin(race_sel))

    out = {'overall': overall, 'macro_f1': macro_f1,
           'macro_race_nr': macro_race, 'worst_race_nr': worst_race, 'gap_race_nr': gap_race,
           'macro_age_nr': macro_age, 'worst_age_nr': worst_age, 'gap_age_nr': gap_age,
           'macro_gen_nr': macro_gen, 'worst_gen_nr': worst_gen, 'gap_gen_nr': gap_gen,
           'dp_race_neutral': dp_race}
    for i, v in enumerate(race_nr): out[f'race{i}_nr'] = v
    for i, v in enumerate(age_nr): out[f'age{i}_nr'] = v
    for i, v in enumerate(gen_nr): out[f'gen{i}_nr'] = v
    return out


def train_one(mode, seed, train_d, val_d, race_counts, gender_counts, age_counts, class_w,
              s1=15, s2=40, s3=20, smooth_lambda=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Head(mode, race_counts, gender_counts, age_counts).to(device)
    n = train_d['features'].shape[0]; bs = 256

    def emb_params():
        ps = []
        for name in ['race_emb', 'gender_emb', 'age_emb']:
            if hasattr(model, name): ps += list(getattr(model, name).parameters())
        return ps

    def aux_params():
        ps = []
        for name in ['aux_race', 'aux_gender', 'aux_age']:
            if hasattr(model, name): ps += list(getattr(model, name).parameters())
        return ps

    mlp_params = list(model.mlp.parameters())

    if mode != 'none':
        opt1 = torch.optim.AdamW(emb_params() + aux_params(), lr=1e-3, weight_decay=0.05)
        for _ in range(s1):
            model.train(); perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; opt1.zero_grad()
                loss = model.aux_losses(train_d['race'][idx], train_d['gender'][idx], train_d['age'][idx])
                loss = loss + smooth_lambda * model.age_smoothness()
                loss.backward(); opt1.step()
        for p in emb_params(): p.requires_grad_(False)

    opt2 = torch.optim.AdamW(mlp_params, lr=1e-3, weight_decay=0.05)
    for _ in range(s2):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt2.zero_grad()
            out = model(train_d['features'][idx], train_d['race'][idx], train_d['gender'][idx], train_d['age'][idx])
            F.cross_entropy(out, train_d['emotion'][idx], weight=class_w).backward()
            opt2.step()

    if mode != 'none':
        for p in emb_params(): p.requires_grad_(True)
        opt3 = torch.optim.AdamW(emb_params() + mlp_params, lr=1e-4, weight_decay=0.05)
    else:
        opt3 = torch.optim.AdamW(mlp_params, lr=1e-4, weight_decay=0.05)

    best_val, best_state = -1, None
    for _ in range(s3):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt3.zero_grad()
            out = model(train_d['features'][idx], train_d['race'][idx], train_d['gender'][idx], train_d['age'][idx])
            loss = F.cross_entropy(out, train_d['emotion'][idx], weight=class_w)
            if mode != 'none':
                loss = loss + smooth_lambda * model.age_smoothness()
            loss.backward(); opt3.step()
        v = compute_metrics(model, val_d)['overall']
        if v > best_val:
            best_val = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def run_full_mode_sweep(train_d, val_d, test_d, race_counts, gender_counts, age_counts, class_w, n_seeds=20):
    MODES = ['none', 'hard_fine', 'hard_best', 'shrinkage']
    records = []
    start = time.time()
    for mode in MODES:
        for seed in range(n_seeds):
            model = train_one(mode, seed, train_d, val_d, race_counts, gender_counts, age_counts, class_w)
            for split_name, d in [('val', val_d), ('test', test_d)]:
                m = compute_metrics(model, d)
                m.update({'mode': mode, 'seed': seed, 'split': split_name})
                if mode == 'shrinkage':
                    m['tau_race'] = torch.exp(model.race_emb.log_tau).item()
                    m['tau_gender'] = torch.exp(model.gender_emb.log_tau).item()
                    m['tau_age'] = torch.exp(model.age_emb.log_tau).item()
                records.append(m)
        print(f"{mode} done ({n_seeds} seeds), {time.time() - start:.0f}s elapsed")

    results_df = pd.DataFrame(records)
    results_df.to_csv(POSTER_VAR_CACHE + 'grouping_results_20seed.csv', index=False)
    print(f"\nAll done in {time.time() - start:.0f}s. Saved {len(results_df)} rows.")

    print("\nTest-split means by model:")
    test_means = results_df[results_df.split == 'test'].groupby('mode')[
        ['overall', 'macro_f1', 'macro_race_nr', 'worst_race_nr', 'gap_race_nr',
         'macro_age_nr', 'worst_age_nr', 'gap_age_nr', 'dp_race_neutral']].mean()
    print(test_means.to_string())
    return results_df


# ============================================================
# 4. Statistical comparison of the sweep results
# ============================================================
def paired_compare(df, metric, mode_a, mode_b):
    a = df[df['mode'] == mode_a].sort_values('seed')[metric].values
    b = df[df['mode'] == mode_b].sort_values('seed')[metric].values
    diff = a - b
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_p = np.nan
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    ci = (diff.mean() - 1.96 * se, diff.mean() + 1.96 * se)
    return {'metric': metric, 'mean_a': a.mean(), 'mean_b': b.mean(),
            'mean_diff': diff.mean(), 'cohen_d': d,
            'ci_low': ci[0], 'ci_high': ci[1], 't_p': t_p, 'wilcoxon_p': w_p}


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m)
    prev = 0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        prev = max(prev, val); adj[idx] = min(prev, 1.0)
    return adj


def run_grouping_stats():
    df = pd.read_csv(POSTER_VAR_CACHE + 'grouping_results_20seed.csv')
    test = df[df.split == 'test'].copy()
    metrics = ['overall', 'macro_f1', 'macro_race_nr', 'worst_race_nr', 'gap_race_nr',
               'macro_age_nr', 'worst_age_nr', 'gap_age_nr', 'dp_race_neutral']

    print("=== PRIMARY: shrinkage vs unconditioned (none), test split, 20 seeds ===")
    rows = [paired_compare(test, m, 'shrinkage', 'none') for m in metrics]
    pv = [r['t_p'] for r in rows]
    adj = holm(pv)
    for r, a in zip(rows, adj): r['holm_t_p'] = a
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))

    print("\n\n=== SECONDARY: shrinkage vs hard_fine (does adaptive grouping beat fixed fine grouping?) ===")
    rows2 = [paired_compare(test, m, 'shrinkage', 'hard_fine') for m in metrics]
    pv2 = [r['t_p'] for r in rows2]
    adj2 = holm(pv2)
    for r, a in zip(rows2, adj2): r['holm_t_p'] = a
    print(pd.DataFrame(rows2).round(4).to_string(index=False))

    sh = df[(df['mode'] == 'shrinkage') & (df['split'] == 'test')]
    print("\n\n=== Learned shrinkage tau (mean across seeds) ===")
    print(f"race:   {sh['tau_race'].mean():.1f} (+/- {sh['tau_race'].std():.1f})")
    print(f"gender: {sh['tau_gender'].mean():.1f} (+/- {sh['tau_gender'].std():.1f})")
    print(f"age:    {sh['tau_age'].mean():.1f} (+/- {sh['tau_age'].std():.1f})")
    return res


# ============================================================
# 5. Sparsity sweep (INACTIVE in the source notebook — preserved as-is)
# ============================================================
#
# def cap_marginal(seed, target):
#     """Cap each demographic attribute independently at `target` samples per group.
#     Returns a boolean keep-mask over the training set."""
#     g = torch.Generator().manual_seed(seed)
#     n = train_d['features'].shape[0]
#     keep = torch.ones(n, dtype=torch.bool)
#     for axis, n_groups in [('race', N_RACE), ('gender', N_GENDER), ('age', N_AGE)]:
#         labels = train_d[axis].cpu()
#         for grp in range(n_groups):
#             idx = torch.where((labels == grp) & keep.cpu())[0]
#             if len(idx) > target:
#                 perm = idx[torch.randperm(len(idx), generator=g)]
#                 drop = perm[target:]
#                 keep[drop] = False
#     return keep.to(device)
#
# def cap_intersectional(seed, target):
#     """Cap each race x gender x age cell at `target` samples. Boolean keep-mask."""
#     g = torch.Generator().manual_seed(seed)
#     n = train_d['features'].shape[0]
#     keep = torch.zeros(n, dtype=torch.bool)
#     r, ge, a = train_d['race'].cpu(), train_d['gender'].cpu(), train_d['age'].cpu()
#     for ri in range(N_RACE):
#         for gi in range(N_GENDER):
#             for ai in range(N_AGE):
#                 idx = torch.where((r == ri) & (ge == gi) & (a == ai))[0]
#                 if len(idx) == 0:
#                     continue
#                 if len(idx) > target:
#                     idx = idx[torch.randperm(len(idx), generator=g)][:target]
#                 keep[idx] = True
#     return keep.to(device)
#
# def train_one_subsampled(mode, seed, keep_mask, s1=15, s2=40, s3=20, smooth_lambda=1e-3):
#     """Same three-stage protocol as train_one, but only trains on rows where
#     keep_mask is True. Evaluation still uses the full untouched val/test sets."""
#     torch.manual_seed(seed); np.random.seed(seed)
#     model = Head(mode).to(device)
#     idx_all = torch.where(keep_mask)[0]
#     feats = train_d['features'][idx_all]
#     emo   = train_d['emotion'][idx_all]
#     race  = train_d['race'][idx_all]
#     gen   = train_d['gender'][idx_all]
#     age   = train_d['age'][idx_all]
#     n = feats.shape[0]; bs = 256
#     ec = torch.bincount(emo, minlength=7).float()
#     cw = (ec.sum() / (7 * ec.clamp(min=1))).to(device)
#     # ... (same 3-stage training loop as train_one, operating on the
#     #      subsampled feats/emo/race/gen/age tensors) ...
#     return model
#
# MODES = ['none', 'hard_fine', 'hard_best', 'shrinkage']
# TARGETS = [25, 50, 100, 200, 400, None]   # None = uncapped control
# SWEEP_SEEDS = 10
# REGIMES = {'marginal': cap_marginal, 'intersectional': cap_intersectional}
# sweep_records = []
# start = time.time()
# for regime_name, cap_fn in REGIMES.items():
#     for target in TARGETS:
#         for mode in MODES:
#             for seed in range(SWEEP_SEEDS):
#                 if target is None:
#                     keep = torch.ones(train_d['features'].shape[0], dtype=torch.bool)
#                     kept_n = keep.sum().item()
#                 else:
#                     keep = cap_fn(seed, target)
#                     kept_n = keep.sum().item()
#                 model = train_one_subsampled(mode, seed, keep)
#                 m = compute_metrics(model, test_d)
#                 m.update({'regime': regime_name, 'target': (target if target else 99999),
#                           'mode': mode, 'seed': seed, 'kept_n': kept_n})
#                 if mode == 'shrinkage':
#                     m['tau_race']   = torch.exp(model.race_emb.log_tau).item()
#                     m['tau_gender'] = torch.exp(model.gender_emb.log_tau).item()
#                     m['tau_age']    = torch.exp(model.age_emb.log_tau).item()
#                 sweep_records.append(m)
#         elapsed = time.time() - start
#         tn = 'full' if target is None else target
#         print(f"{regime_name} target={tn} done, {elapsed:.0f}s elapsed")
# sweep_df = pd.DataFrame(sweep_records)
# sweep_df.to_csv(POSTER_VAR_CACHE + 'sparsity_sweep_10seed.csv', index=False)
# print(f"\nSweep complete: {len(sweep_df)} rows in {time.time()-start:.0f}s")


def run_sparsity_sweep_analysis():
    """Analysis of the (pre-generated) sparsity_sweep_10seed.csv."""
    sweep_df = pd.read_csv(POSTER_VAR_CACHE + 'sparsity_sweep_10seed.csv')
    metrics = ['overall', 'macro_race_nr', 'worst_race_nr', 'macro_age_nr']

    def regime_table(regime):
        sub = sweep_df[sweep_df.regime == regime]
        rows = []
        for target in sorted(sub.target.unique()):
            t = sub[sub.target == target]
            means = t.groupby('mode')[metrics].mean()
            row = {'target': ('full' if target == 99999 else target),
                   'kept_n': int(t['kept_n'].mean())}
            for metric in metrics:
                sh = means.loc['shrinkage', metric]
                hf = means.loc['hard_fine', metric]
                nn_ = means.loc['none', metric]
                row[f'{metric}_shrink'] = round(sh, 4)
                row[f'{metric}_vs_hardfine'] = round(sh - hf, 4)
                row[f'{metric}_vs_none'] = round(sh - nn_, 4)
            rows.append(row)
        return pd.DataFrame(rows)

    for regime in ['marginal', 'intersectional']:
        print(f"\n{'=' * 70}\n{regime.upper()} REGIME\n{'=' * 70}")
        tbl = regime_table(regime)
        print("\nShrinkage minus hard_fine (positive = shrinkage wins):")
        cols = ['target', 'kept_n'] + [f'{m}_vs_hardfine' for m in metrics]
        print(tbl[cols].to_string(index=False))
        print("\nShrinkage minus none (positive = conditioning helps):")
        cols = ['target', 'kept_n'] + [f'{m}_vs_none' for m in metrics]
        print(tbl[cols].to_string(index=False))

    print(f"\n{'=' * 70}\nLEARNED TAU vs SPARSITY (shrinkage model, mean across seeds)\n{'=' * 70}")
    tau_tbl = sweep_df[sweep_df['mode'] == 'shrinkage'].groupby(['regime', 'target'])[
        ['tau_race', 'tau_gender', 'tau_age']].mean().round(1)
    print(tau_tbl.to_string())
    return sweep_df


if __name__ == '__main__':
    train_d, val_d, test_d, race_counts, gender_counts, age_counts, class_w = load_full_features()
    run_train_staged_demo(train_d, val_d, test_d, race_counts, gender_counts, age_counts, class_w)
    run_full_mode_sweep(train_d, val_d, test_d, race_counts, gender_counts, age_counts, class_w)
    run_grouping_stats()
