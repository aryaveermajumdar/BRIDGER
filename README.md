# BRIDGER: Bias Reduction In Demographic Group Emotion Recognition

Code for training, auditing, and statistically evaluating demographic-conditioned facial emotion recognition (FER) models on RAF-DB and AffectNet+, built on a POSTER-Var backbone (IR50 + MobileFaceNet + cross-attention pyramid fusion).

This repository accompanies the paper submitted to *IEEE Transactions on Affective Computing*. It contains the full pipeline used to produce every model, figure, and statistical result reported there: dataset preparation, demographic annotation parsing, model training, baseline reimplementations, fairness auditing, calibration analysis, and significance testing.

## Overview

Standard FER models are typically trained and reported as a single global accuracy number, which can mask large disparities in performance across demographic subgroups (race, gender, age). BRIDGER investigates and addresses this at three levels:

1. **Architecture**: demographic-conditioned shrinkage embeddings that let the classification head adapt to subgroup identity while regularizing toward the global embedding for sparsely represented groups, avoiding overfitting on small subgroups.
2. **Training objective**: a joint multitask formulation that trades off emotion classification loss against an auxiliary demographic-prediction loss, swept across a range of weighting values (lambda) to trace the accuracy-fairness frontier rather than reporting a single operating point.
3. **Evaluation**: fairness is audited from multiple independent angles, per-group accuracy, neutral-class recall and selection rate, calibration (expected calibration error), and demographic decodability from learned representations via multi-seed linear probes, so that a model cannot look fair on one metric while hiding disparity on another.

Two baseline debiasing approaches from prior work (attribute-aware conditioning and adversarial gradient-reversal, following Xu et al.) are reimplemented for direct, apples-to-apples comparison against the proposed shrinkage-conditioned and joint multitask models.

All pairwise model comparisons are backed by paired bootstrap confidence intervals and permutation significance tests, with Holm-Bonferroni correction applied across the full family of comparisons.

## Datasets

| Dataset | Role | Scale | Notes |
|---|---|---|---|
| RAF-DB (basic, aligned) | Primary training and evaluation set | ~15,300 aligned face images, 7 basic expressions | Demographic labels (race, gender, age) parsed from the dataset's manual annotation files |
| AffectNet+ | Held-out generalization check | ~114,000 images after filtering to the 7 basic expressions shared with RAF-DB | Contempt is either merged into Disgust or dropped depending on the run, see the pipeline script for the exact protocol used in each |

Both datasets are third-party resources with their own access terms. This repository does **not** redistribute any images or labels. Request access from the original dataset maintainers, then point the path variables described in "Configuration" below at your local copies.

## Repository structure

| File | Contents |
|---|---|
| `bridger_rafdb_pipeline.py` | End-to-end RAF-DB pipeline: environment setup, POSTER-Var backbone and IR50 weights download, demographic annotation parsing, feature extraction, the core model classes (`PlainEmbedding`, `ShrinkageEmbedding`, `Head`, `DemogClf`), staged training, demographic decodability probing, the joint multitask model and its lambda sweep, and the paired bootstrap/permutation significance testing utilities. |
| `bridger_rafdb_analysis.py` | Statistical audit and figures for the RAF-DB models: baseline reimplementations (`XuAttributeAware`, `XuAdversarial`), the full paired significance-testing suite, multi-seed decodability probes, calibration analysis (ECE and reliability bins), confusion matrices, and all architecture/results diagrams. |
| `bridger_affectnet_pipeline.py` | Generalization check on AffectNet+: dataset assembly and age-binning, the POSTER-Var training loop with a from-scratch SAM (sharpness-aware minimization) optimizer, EMA weight averaging, mixup/cutmix augmentation, and cosine LR scheduling with warmup. Contains two successive training runs (an initial run and a fresh-start rerun with fast data loading and full checkpoint-resume support for long sessions). |
| `bridger_affectnet_analysis.py` | Statistical audit and figures for the AffectNet+ models, mirroring the RAF-DB analysis: feature extraction hooks, demographic probes, multi-seed decodability, and diagram generation. |

Each `*_pipeline.py` script produces the trained checkpoints and cached features consumed by its corresponding `*_analysis.py` script. Run the pipeline for a dataset before running its analysis.

## Requirements

- Python 3.9+
- A CUDA-capable GPU is strongly recommended; all four scripts were developed and run on Google Colab GPU runtimes.

Core dependencies:

```
torch
torchvision
numpy
pandas
scipy
scikit-learn
matplotlib
tqdm
Pillow
thop
```

Install with:

```bash
pip install torch torchvision numpy pandas scipy scikit-learn matplotlib tqdm Pillow thop
```

The POSTER-Var backbone code and pretrained IR50 weights are fetched automatically at runtime from the original authors' repository, no manual download is required.

## Important: these scripts are Colab exports

Each script is a verbatim export of the Google Colab notebook used for the corresponding stage of the project. They still contain Colab-specific constructs:

- `from google.colab import drive` and `drive.mount(...)` calls
- `!pip install ...` magic commands (not valid outside IPython/Colab)
- Hardcoded paths under `/content/drive/MyDrive/Colab Notebooks/...`

To run outside Colab:

1. Remove or comment out the `drive.mount(...)` call and any `!pip install` lines. Install dependencies from the requirements list above instead.
2. Replace the hardcoded `/content/drive/MyDrive/Colab Notebooks/...` paths (search each script for `RAFDB_BASE_CACHE`, `AFFECTNET_RAW_DIR`, `POSTER_VAR_CACHE`, and similar variables near the top of each file) with local paths to your dataset, checkpoint, and cache directories.
3. The AffectNet+ pipeline script ends with an `IPython.display.Audio` completion alert. This is Colab/Jupyter-only and can be safely removed for a plain Python run.

No modeling, training, or evaluation logic was altered from the original notebooks, only the setup boilerplate above needs adjusting to run locally.

## Usage

Run each dataset's pipeline script before its analysis script:

```bash
# RAF-DB
python bridger_rafdb_pipeline.py
python bridger_rafdb_analysis.py

# AffectNet+
python bridger_affectnet_pipeline.py
python bridger_affectnet_analysis.py
```

Each script is organized as a sequence of top-to-bottom cells (preserved from the original notebooks) rather than a CLI with argument parsing. To rerun only part of a pipeline, open the script in an interactive environment (Jupyter, VS Code interactive window, or Colab) and execute the relevant section, most sections check for existing cached outputs and skip redundant work automatically.

## Method summary

- **Backbone**: POSTER-Var (IR50 + MobileFaceNet cross-attention pyramid), initialized from pretrained weights and fine-tuned per dataset.
- **Demographic-conditioned heads**: `ShrinkageEmbedding` shrinks each demographic subgroup's embedding toward the global embedding in proportion to how underrepresented that subgroup is in the training set, reducing overfitting on small groups.
- **Joint multitask training**: `JointModel` jointly predicts emotion and demographic attributes; `run_lambda_sweep` trains across a range of loss-weighting values to trace the resulting accuracy-fairness tradeoff curve rather than a single point.
- **Baselines**: `XuAttributeAware` (attribute-conditioned) and `XuAdversarial` (gradient-reversal adversarial debiasing) are reimplemented for direct comparison.
- **Fairness metrics**: per-group accuracy, neutral-class recall and selection rate by subgroup, and demographic decodability (linear-probe accuracy at predicting race/gender/age from the learned representation, averaged over multiple random seeds).
- **Calibration**: expected calibration error (ECE) with reliability diagrams.
- **Statistical testing**: paired bootstrap confidence intervals and permutation tests for every head-to-head model comparison, with Holm-Bonferroni correction applied across the full family of comparisons to control the false discovery rate.

## Reproducibility notes

- Decodability probes and per-group results are computed across multiple random seeds (5 to 10 depending on the script) and reported as distributions rather than single-seed point estimates.
- Training uses early stopping on validation accuracy with a fixed patience window; the best checkpoint (by validation accuracy) is saved for downstream analysis.
- The AffectNet+ pipeline additionally supports resuming from a full training-state checkpoint, useful for reproducing a run interrupted by a Colab session timeout.

## Results

Full quantitative results, ablation tables, and statistical comparisons are reported in the accompanying paper. Running the analysis scripts regenerates the underlying figures and result tables locally as CSV and PNG files in your configured cache directory.

## Acknowledgments
- Backbone architecture and pretrained weights from the [POSTER-Var](https://github.com/lg2578/poster-var) repository.
- RAF-DB and AffectNet+ dataset creators and maintainers.

## License

This code is intended for release under the MIT License. Add a `LICENSE` file to the repository root with the standard MIT text and your name and year to formalize this. The datasets used with this code carry their own separate licenses and access terms, set by their respective creators, and are not covered by this repository's license.

## Contact

Aryaveer Majumdar
aryaveermajumdar@gmail.com

Questions, issues, and pull requests are welcome via the repository's issue tracker.x
