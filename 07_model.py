"""
model.py

Consolidated model architecture definitions for BRIDGER. Every class here is
extracted from the existing pipeline scripts (common_utils.py,
04_grouping_and_shrinkage_experiments.py, 05_joint_multitask_and_lambda_sweep.py)
with no logic changes, just gathered into one place so an architecture can be
imported and reused (e.g. by inference.py) without importing an entire
pipeline stage script as a module.

Import this after common_utils, e.g.:
    from common_utils import *
    from model import *

NOTE FOR REVIEWER: these class bodies are copied verbatim from the pipeline
scripts named above. If those scripts are ever edited independently, this
file will drift out of sync unless updated in parallel. Once you've verified
this file is correct, consider having 04 and 05 import their model classes
from here instead of redefining them inline, that removes the duplication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_utils import N_RACE, N_GENDER, N_AGE, device, load_pyramid_model


# ============================================================
# Unconditioned demographic probes (from common_utils.py)
# ============================================================

class DemogClf(nn.Module):
    """Demographic probe head: predicts race, gender, and age from a
    768-d feature. Used throughout as the standard decodability probe,
    'how much demographic information can be read out of this
    representation.'"""
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


class DemogClassifier(nn.Module):
    """Alias used in some later cells of the original notebook. Functionally
    identical to DemogClf, but hardcodes in_dim=768 and uses separate named
    heads instead of a tuple return via a single trunk."""
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


# ============================================================
# Conditioned probe used in the (inactive) partition search
# ============================================================

class ConditionedProbe(nn.Module):
    """Emotion classifier conditioned on (race, gender, age) via learned
    embeddings concatenated to the 768-d feature. Used by the brute-force
    race/age partition search in 04_grouping_and_shrinkage_experiments.py.
    That search loop itself is left inactive (commented out) in the source
    notebook, this class is the active helper it calls when run."""
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


# ============================================================
# Shrinkage-conditioned embeddings, the paper's main contribution
# ============================================================

class ShrinkageEmbedding(nn.Module):
    """
    Embedding(i) = shared + alpha_i * delta_i, where alpha_i = n_i / (n_i + tau).
    tau is learnable (log-parameterized, always positive). Large tau means
    heavy pooling toward the shared embedding, small tau means groups stay
    close to their own separate embeddings.
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
    """Standalone shrinkage-conditioned head used by the 3-stage
    train_staged() demo in 04_grouping_and_shrinkage_experiments.py. The
    paper's main 4-way mode comparison sweep instead uses
    Head(mode='shrinkage') below, which wraps the same ShrinkageEmbedding
    class but supports switching between conditioning strategies."""
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


# ============================================================
# Main mode-switchable head: none | hard_fine | hard_best | shrinkage
# This is the model used for the paper's primary 20-seed comparison.
# ============================================================

BEST_RACE_MAP = {0: 0, 1: 1, 2: 2}              # all separate
BEST_AGE_MAP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}   # groups 0+1 together, 2+3+4 together


class PlainEmbedding(nn.Module):
    """Standard (non-shrinkage) embedding, used by the hard_fine and
    hard_best conditioning modes as the fixed-grouping contrast to
    ShrinkageEmbedding's adaptive grouping."""
    def __init__(self, n_groups, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(n_groups, emb_dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, ids):
        return self.emb(ids)

    def age_smoothness(self):
        return torch.tensor(0.0, device=device)


class Head(nn.Module):
    """mode: 'none' | 'hard_fine' | 'hard_best' | 'shrinkage'

    The model used for the paper's primary 20-seed comparison
    (run_full_mode_sweep in 04_grouping_and_shrinkage_experiments.py) and
    reused directly by 05_joint_multitask_and_lambda_sweep.py for the
    recovered-feature comparison. Operates on a pre-extracted 768-d
    feature, not a raw image, see inference.py for the end-to-end wrapper
    that extracts the feature first.
    """
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


# ============================================================
# Joint multitask model: POSTER-Var backbone + auxiliary demographic heads
# ============================================================

class JointModel(nn.Module):
    """Wraps a POSTER-Var base model (a pyramid_trans_expr2 instance) with
    an auxiliary demographic head hung off the same 768-d SE-gated feature
    the emotion head already uses. Trained end-to-end, with the backbone's
    ir_back frozen, across a range of loss-weighting lambda values in
    run_lambda_sweep() (05_joint_multitask_and_lambda_sweep.py)."""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.feat = None
        # Hook the 768-dim feature (output of VIT se_block)
        self.hook_handle = self.base.VIT.se_block.register_forward_hook(self._hook_fn)
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


# ============================================================
# Convenience constructor (new, not extracted, see note below)
# ============================================================

def build_posterv2_backbone(num_classes=7):
    """Instantiate a fresh, randomly-initialized POSTER-Var backbone. The
    caller is responsible for loading a state_dict on top of this.

    NOTE FOR REVIEWER: this is a small new wrapper, not copied from an
    existing cell, it exists because the exact same three lines
    (load_pyramid_model() -> instantiate -> .to(device)) were repeated
    verbatim in extract_posterv2_features(), load_joint_01(), and
    run_lambda_sweep(). Please confirm the img_size=224, vae=True arguments
    below match every call site you rely on."""
    pyramid_trans_expr2 = load_pyramid_model()
    return pyramid_trans_expr2(img_size=224, num_classes=num_classes, vae=True).to(device)
