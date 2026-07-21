# BRIDGER: Bias Reduction In Demographic Group Emotion Recognition

Code for training, auditing, and statistically evaluating a demographic-conditioned facial emotion recognition (FER) model on RAF-DB, built on a POSTER-Var backbone (IR50 + MobileFaceNet + cross-attention pyramid fusion).

This repository contains the full pipeline behind the project: dataset preparation, demographic annotation parsing, model training, decodability auditing, and significance testing, plus the code used to produce every figure and statistical result.

## Overview

Standard FER models are typically trained and reported as a single global accuracy number, which can mask large disparities in performance across demographic subgroups (race, gender, age). BRIDGER investigates and addresses this at two levels:

1. **Backbone level**: an auxiliary demographic-preservation objective, trained jointly with the emotion classification loss during POSTER-Var fine-tuning, that recovers minority race and age decodability the plain fine-tuned backbone otherwise collapses. Swept across a range of loss-weighting values (lambda) rather than reported at a single operating point.
2. **Head level**: demographic-conditioned shrinkage embeddings that let the classification head adapt to subgroup identity, while regularizing toward the global embedding for sparsely represented groups, so that conditioning does not just overfit small subgroups. Compared directly against fixed (non-adaptive) grouping and against no conditioning at all across a 20-seed sweep.

Fairness is audited from several angles: per-group accuracy, Neutral-class recall and the gap in Neutral recall and Neutral selection rate across race/gender/age subgroups, and demographic decodability from the learned representation via linear probes (both an unconditioned probe and a paired comparison against the pristine, pre-expression-training backbone). Model comparisons are backed by paired t-tests, Wilcoxon signed-rank tests, and Holm-corrected p-values; the head-to-head comparison on the recovered (lambda=0.1) features additionally uses a full battery of a Shapiro normality check, a permutation test, and bootstrap confidence intervals, with Holm-Bonferroni correction across all three p-values.

## Dataset

| Dataset | Role | Scale | Notes |
|---|---|---|---|
| RAF-DB (basic, aligned) | Training and evaluation set | roughly 15,300 aligned face images, 7 basic expressions | Demographic labels (race, gender, age) parsed from the dataset's manual annotation files |

RAF-DB is a third-party resource with its own access terms. This repository does **not** redistribute any images or labels. Request access from the original dataset maintainers, then point the path variables in `common_utils.py` at your local copy.

## Repository structure

| File | Contents |
|---|---|
| `common_utils.py` | Shared paths and constants, image transforms, the `RafDbDataset` / `RafDbDemogDataset` classes, the demographic probe heads `DemogClf` / `DemogClassifier`, a lazy loader for the POSTER-Var backbone class, and small evaluation helpers (`evaluate_model`, `check_checkpoint`, `per_group_accuracy`). Import this first, every other script depends on it. |
| `01_setup_and_prepare_data.py` | Mounts Drive, downloads the POSTER-Var repository and its IR50/MobileFaceNet pretrained weights, patches hardcoded paths in that repository, copies RAF-DB images locally and builds train/val/test label CSVs, verifies and restores checkpoints from a backup folder, parses RAF-DB's manual demographic annotations (gender, race, age), and attaches those demographics to the cached POSTER-Var features. |
| `02_feature_extraction.py` | Extracts the 768-d SE-gated POSTER-Var feature (EMA checkpoint) for train/val/test, extracts pristine IR50 features (the face-recognition backbone before any expression training), and extracts the IR50 stream from inside the fine-tuned POSTER-Var model. These three feature sets feed `03`. |
| `03_decodability_analysis.py` | A 3-point race decodability comparison across the pipeline (pristine IR50, IR50 stream inside POSTER-Var, final POSTER-Var feature), weighted and unweighted demographic classifiers with per-group accuracy and confusion matrices, a 10-seed paired comparison of pristine-IR50 versus POSTER-Var decodability, and a linear probe of emotion accuracy directly on pristine IR50 features for reference. |
| `04_grouping_and_shrinkage_experiments.py` | The `ShrinkageEmbedding` / `ShrinkageConditionedHead` classes and their 3-stage training protocol, the main mode-switchable `Head` (`none` / `hard_fine` / `hard_best` / `shrinkage`) used for the primary 20-seed comparison sweep, paired t-test/Wilcoxon/Holm-corrected statistical comparison of that sweep, and sparsity-sweep analysis helpers. Also contains an inactive (preserved, not executed) brute-force race/age partition search. |
| `05_joint_multitask_and_lambda_sweep.py` | `JointModel` (the POSTER-Var backbone plus auxiliary race/gender/age heads), `run_lambda_sweep()` which trains it across several loss-weighting lambda values, evaluation of the saved lambda=0.1 checkpoint on the test set overall and per race group, re-extraction of the 768-d feature from that recovered model with demographics re-attached, and a `none` vs `hard_fine` conditioning comparison on the recovered features with the full statistical battery described above. Imports `Head` and related functions from `04` at runtime via `importlib`, see Usage below. |
| `06_figures.py` | The three project figures: an architecture schematic, a bar chart of race decodability across the pipeline stages, and a bar chart of the conditioning effect (macro race Neutral recall, no conditioning vs `hard_fine`, collapsed vs recovered features). |
| `model.py` | Not yet in the repository as of this README. Consolidates the model class definitions above (`DemogClf`, `DemogClassifier`, `ShrinkageEmbedding`, `ShrinkageConditionedHead`, `Head`, `JointModel`, and related helpers) into one importable file, pulled from `common_utils.py`, `04`, and `05` with no logic changes. |
| `inference.py` | Not yet in the repository as of this README. New code, not part of the original notebooks: lets a reader run the trained model on a single image or a folder of images without executing the full pipeline. |

## Requirements

- Python 3.9+
- A CUDA-capable GPU is strongly recommended.

Core dependencies, confirmed against actual imports in this repository:

```
torch
torchvision
numpy
pandas
scipy
scikit-learn
matplotlib
Pillow
requests
gdown
```

Install with:

```bash
pip install torch torchvision numpy pandas scipy scikit-learn matplotlib Pillow requests gdown
```

The POSTER-Var backbone code and pretrained IR50/MobileFaceNet weights are fetched automatically at runtime by `01_setup_and_prepare_data.py`, no manual download is required.

## Important: this pipeline was built and run on Colab

`01_setup_and_prepare_data.py` mounts Google Drive (`from google.colab import drive; drive.mount(...)`) and restores checkpoints from a hardcoded `/content/drive/MyDrive/new_saves/` path. `common_utils.py` also hardcodes its cache paths under `/content/drive/MyDrive/Colab Notebooks/...`.

To run outside Colab:

1. Remove or comment out the `drive.mount(...)` call in `01_setup_and_prepare_data.py`.
2. Replace the hardcoded `/content/drive/MyDrive/...` paths, in `common_utils.py` (`RAFDB_BASE_CACHE`, `POSTER_VAR_CACHE`) and in `01_setup_and_prepare_data.py`'s checkpoint-restore step, with local paths to your dataset, checkpoint, and cache directories.

No modeling, training, or evaluation logic was altered from the original notebook cells, only this setup boilerplate needs adjusting to run locally.

## Usage

Run in numeric order, each script depends on cached outputs from the one before it:

```bash
python 01_setup_and_prepare_data.py
python 02_feature_extraction.py
python 03_decodability_analysis.py
python 04_grouping_and_shrinkage_experiments.py
python 05_joint_multitask_and_lambda_sweep.py
python 06_figures.py
```

Two things worth knowing before you run this:

- `05_joint_multitask_and_lambda_sweep.py` imports `Head`, `train_one`, and `compute_metrics` from `04_grouping_and_shrinkage_experiments.py` at runtime via `importlib.import_module('04_grouping_and_shrinkage_experiments')`, since a leading digit makes that filename an invalid plain `import` target. Run scripts from the repository root so `04`'s file is importable this way.
- The second and third figures in `06_figures.py` (`make_decodability_pipeline_figure`, `make_conditioning_effect_figure`) plot literal, hardcoded result values copied in from earlier analysis output, they are not recomputed from the cached CSVs at figure-generation time. If you rerun the pipeline with different data or seeds, update those hardcoded arrays yourself before regenerating the figures.

Each script is organized as a sequence of top-to-bottom cells preserved from the original notebooks, most functions check for existing cached outputs and skip redundant work automatically.

## Method summary

- **Backbone**: POSTER-Var (IR50 + MobileFaceNet cross-attention pyramid), fine-tuned on RAF-DB from pretrained weights.
- **Backbone-level fix**: `JointModel` adds an auxiliary demographic-prediction head off the same 768-d feature the emotion head uses, trained jointly with a loss-weighting lambda swept across several values in `run_lambda_sweep()`. This recovers race and age decodability that collapses in the plain fine-tuned backbone, before any head-level conditioning is applied.
- **Head-level conditioning**: `ShrinkageEmbedding` shrinks each demographic subgroup's embedding toward a shared global embedding in proportion to how underrepresented that subgroup is in the training set, with a learnable per-attribute shrinkage strength. Compared against fixed fine-grained grouping (`hard_fine`), a manually chosen coarser grouping (`hard_best`), and no conditioning (`none`) across a 20-seed sweep.
- **Fairness metrics**: per-group accuracy, Neutral-class recall and its gap across subgroups, a demographic-parity-style gap in Neutral selection rate across race groups, and demographic decodability (linear-probe accuracy at predicting race/gender/age from the learned representation).
- **Statistical testing**: paired t-tests, Wilcoxon signed-rank tests, and Holm-corrected p-values for the main 20-seed conditioning sweep (`04`); a fuller battery of a Shapiro normality check, a permutation test, and bootstrap confidence intervals, with Holm-Bonferroni correction across all three, for the `hard_fine` vs `none` comparison on the lambda=0.1 recovered features (`05`).

## Reproducibility notes

- Decodability probes and sweep results are computed across multiple random seeds, 5 in the main pipeline decodability comparison, 10 in the paired IR50-vs-POSTER-Var decodability comparison and the lambda=0.1 conditioning comparison, and 20 in the main shrinkage-vs-baseline conditioning sweep, and reported as distributions rather than single-seed point estimates.
- Two of the three figures in `06_figures.py` use hardcoded literal result values rather than recomputing from cached CSVs, see the note under Usage above.

## Results

Full quantitative results and statistical comparisons are written up separately from this repository. Running the analysis scripts regenerates the underlying result tables and figures locally as CSV, PNG, and PDF files in your configured cache directory.

## Citation

If you use this code, please cite:

```bibtex
@misc{bridger,
  title  = {Diagnosing and Reversing Demographic Signal Collapse in Facial Expression Recognition},
  author = {Majumdar, Aryaveer and Vinet, Micah},
  year   = {2026},
  url    = {https://github.com/aryaveermajumdar/BRIDGER}
}
```

## Acknowledgments

- Backbone architecture and pretrained weights from the [POSTER-Var](https://github.com/lg2578/poster-var) repository, licensed under Apache License 2.0. That code is downloaded at runtime rather than redistributed in this repository, see the original repository for its full license terms.
- RAF-DB dataset creators and maintainers.

## License

This repository's original code is released under the MIT License, see `LICENSE`. This covers the code in this repository only. The POSTER-Var backbone it depends on is separately licensed under Apache License 2.0 by its original authors, and the RAF-DB dataset carries its own separate access terms set by its creators. Neither is covered by this repository's MIT license.

## Contact

Aryaveer Majumdar <aryaveermajumdar@gmail.com>

Questions, issues, and pull requests are welcome via the repository's issue tracker.
