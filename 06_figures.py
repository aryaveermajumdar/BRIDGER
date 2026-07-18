"""
06_figures.py

Consolidates the three matplotlib figure-generation cells from the notebook:

  1. fig1_architecture — schematic of the B.R.I.D.G.E.R. backbone-level fix
     (auxiliary demographic loss) + staged demographic-conditioned head.
  2. fig2_decodability_pipeline — race decodability bar chart across the
     four pipeline stages (pristine IR50 -> IR50 stream -> collapsed
     POSTER-Var feature -> lambda=0.1 recovered feature).
  3. fig3_conditioning_effect — macro race Neutral recall, no-conditioning
     vs hard_fine conditioning, for collapsed vs recovered features.

Each function saves .png/.pdf to POSTER_VAR_CACHE, matching the originals.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from common_utils import POSTER_VAR_CACHE

plt.rcParams['font.family'] = 'serif'


# ============================================================
# Figure 1: Architecture diagram
# ============================================================
def make_architecture_figure(save=False):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    def box(x, y, w, h, text, color='#E8EEF7', fontsize=9, edge='#333333'):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.05",
                            facecolor=color, edgecolor=edge, linewidth=1.2)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12, color='#333333'))

    # Input / backbone
    box(0.3, 8.2, 1.8, 1.0, "Face Image\n224x224x3", '#F5F5F5')
    box(0.3, 6.6, 1.8, 1.0, "POSTER-Var\n(IR50 + MobileFaceNet\n+ cross-attn fusion)", '#DCE8F5')
    box(0.3, 5.0, 1.8, 1.0, "768-d SE-gated\nCLS-equiv. feature", '#DCE8F5')
    arrow(1.2, 8.2, 1.2, 7.6); arrow(1.2, 6.6, 1.2, 6.0)

    # NEW: demographic preservation objective (lambda 0.1)
    box(2.8, 5.0, 2.2, 1.0, "Aux. demographic\ntrunk + heads\n(race/gender/age)", '#F5E6D3')
    arrow(2.1, 5.5, 2.8, 5.5)
    box(2.8, 3.6, 2.2, 1.0, "L_demog\n(preserves minority\nrace/age signal)", '#F5E6D3')
    arrow(3.9, 5.0, 3.9, 4.6)

    # Demographic-conditioned head
    box(0.3, 3.2, 1.8, 1.0, "Race/Gender/Age\nembeddings\n(learned, 32-d each)", '#DCF5E0')
    arrow(1.2, 5.0, 1.2, 4.2)
    box(0.3, 1.6, 1.8, 1.0, "Concat: 768+96=864", '#DCE8F5')
    box(0.3, 0.2, 1.8, 1.0, "MLP -> 7 emotion\nlogits", '#E8EEF7')
    arrow(1.2, 3.2, 1.2, 2.6); arrow(1.2, 1.6, 1.2, 1.2)

    # Stage boxes
    box(5.6, 7.6, 4.1, 1.0, "Stage 1 (15 ep): embeddings\n+ aux heads; MLP frozen", '#DCE8F5')
    box(5.6, 6.2, 4.1, 1.0, "Stage 2 (40 ep): emotion MLP\nonly; embeddings frozen", '#DCF5E0')
    box(5.6, 4.8, 4.1, 1.0, "Stage 3 (20 ep): joint\nfine-tune, lr=1e-4", '#F0DCF5')
    ax.text(7.65, 3.9,
            "Backbone-level fix (this work):\nauxiliary demographic loss\nduring POSTER-Var training\nrecovers minority race/age\ndecodability before staged\nconditioning begins.",
            ha='center', va='top', fontsize=7, style='italic', color='#555555')

    ax.text(5.0, 9.3, "B.R.I.D.G.E.R. on POSTER-Var: backbone-level demographic preservation\n+ staged demographic-conditioned head",
            ha='center', fontsize=16, weight='bold')

    plt.tight_layout()
    if save:
        plt.savefig(POSTER_VAR_CACHE + 'fig1_architecture.png', dpi=300, bbox_inches='tight')
        plt.savefig(POSTER_VAR_CACHE + 'fig1_architecture.pdf', bbox_inches='tight')
        print("Saved fig1_architecture.png/.pdf")
    plt.show()


# ============================================================
# Figure 2: Race decodability along the pipeline
# ============================================================
def make_decodability_pipeline_figure(save=True):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    stages = ['Pristine IR50\n(pre-expression)', 'IR50 stream\n(inside POSTER-Var)',
              'POSTER-Var final\nfeature (collapsed)', 'Lambda=0.1\n(recovered)']
    white_vals = [0.947, np.nan, 0.996, 0.885]
    black_vals = [0.716, 0.710, 0.024, 0.752]
    asian_vals = [0.667, 0.661, 0.001, 0.754]
    white_err = [0.011, 0, 0.002, 0]
    black_err = [0.031, 0.036, 0.014, 0]
    asian_err = [0.043, 0.016, 0.001, 0]

    x = np.arange(len(stages))
    w = 0.25

    ax.bar(x - w, white_vals, w, yerr=white_err, label='White', color='#B0C4DE')
    ax.bar(x, black_vals, w, yerr=black_err, label='Black', color='#8B5A2B')
    ax.bar(x + w, asian_vals, w, yerr=asian_err, label='Asian', color='#DAA520')

    ax.set_ylabel('Race decodability (classifier accuracy)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(1 / 3, color='gray', linestyle=':', linewidth=1, label='Chance (3-class)')
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.set_title('Minority race decodability collapses during expression training,\n'
                  'and is recovered by an auxiliary preservation objective (lambda=0.1)',
                  fontsize=10.5)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    if save:
        plt.savefig(POSTER_VAR_CACHE + 'fig2_decodability_pipeline.png', dpi=300, bbox_inches='tight')
        plt.savefig(POSTER_VAR_CACHE + 'fig2_decodability_pipeline.pdf', bbox_inches='tight')
        print("Saved fig2_decodability_pipeline.png/.pdf")
    plt.show()


# ============================================================
# Figure 3: Demographic conditioning benefit, collapsed vs recovered
# ============================================================
def make_conditioning_effect_figure(save=True):
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = ['Collapsed features\n(original backbone)', 'Recovered features\n(lambda=0.1)']
    none_vals = [0.8782, 0.8981]
    hf_vals = [0.8944, 0.9055]
    # CI half-widths for hard_fine, from the rigorous battery (lambda 0.1) and prior sweep
    hf_ci = [(0.8944 - 0.8944) + 0.006, (0.9055 - 0.9055) + (0.0107 - 0.0074)]
    none_ci = [0.006, 0.005]

    x = np.arange(len(conditions))
    w = 0.32

    ax.bar(x - w / 2, none_vals, w, label='No conditioning', color='#C9C9C9')
    ax.bar(x + w / 2, hf_vals, w, yerr=[[0, 0], [hf_ci[0], hf_ci[1]]], label='hard_fine conditioning',
           color='#5B84B1', edgecolor='black', linewidth=0.6, capsize=4)

    for i, (n, h) in enumerate(zip(none_vals, hf_vals)):
        ax.annotate(f'+{(h - n) * 100:.2f}pp', xy=(i, max(n, h) + 0.008), ha='center')

    ax.set_ylabel('Macro race Neutral recall', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=9.5)
    ax.set_ylim(0.85, 0.93)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title('Demographic conditioning benefit is stable across feature regimes;\n'
                  'recovering backbone decodability further improves the effect',
                  fontsize=10.5)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    if save:
        plt.savefig(POSTER_VAR_CACHE + 'fig3_conditioning_effect.png', dpi=300, bbox_inches='tight')
        plt.savefig(POSTER_VAR_CACHE + 'fig3_conditioning_effect.pdf', bbox_inches='tight')
        print("Saved fig3_conditioning_effect.png/.pdf")
    plt.show()


if __name__ == '__main__':
    make_architecture_figure(save=False)  # save=False in the source notebook
    make_decodability_pipeline_figure(save=True)
    make_conditioning_effect_figure(save=True)
