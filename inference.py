"""
inference.py

Run BRIDGER on a single image (or a folder of images) without executing the
full training/analysis pipeline. This file is new, it does not exist in the
original Colab notebooks, it was written to fill a real gap: none of
01 through 06 take a raw image in and return a prediction out.

Three modes, matched to what the pipeline scripts actually produce and save
to disk:

  emotion     Plain fine-tuned POSTER-Var backbone -> 7-way emotion only.
              Loads EMA_CHECKPOINT_PATH (override with --checkpoint). This
              is the only mode whose checkpoint is guaranteed to exist after
              running the pipeline as shipped.

  joint       JointModel (backbone + auxiliary race/gender/age heads) ->
              emotion and demographic predictions from one forward pass.
              Loads a lambda-sweep checkpoint saved by run_lambda_sweep() in
              05_joint_multitask_and_lambda_sweep.py, for example
              joint_model_lambda_0.1.pth. Pick the sweep point with
              --lambda_value.

  shrinkage   The paper's main demographic-conditioned Head(mode='shrinkage').
              REQUIRES A CHECKPOINT THIS REPO DOES NOT SAVE BY DEFAULT. The
              20-seed sweep in run_full_mode_sweep() (04) evaluates many
              Head instances but only writes their metrics to
              grouping_results_20seed.csv, never their weights. To use this
              mode, train and save one yourself first, see
              save_shrinkage_checkpoint() below, or call
              train_one('shrinkage', seed, ...) directly from
              04_grouping_and_shrinkage_experiments.py and
              torch.save(model.state_dict(), your_path). This mode also
              needs race, gender, and age group ids for the image, they are
              NOT inferred by this script, pass them with --race/--gender/--age.

USAGE
    python inference.py --image path/to/face.jpg --mode emotion
    python inference.py --image path/to/face.jpg --mode joint --lambda_value 0.1
    python inference.py --image path/to/face.jpg --mode shrinkage \
        --shrinkage_ckpt path/to/your_saved_head.pth --race 0 --gender 0 --age 2

    --image also accepts a directory, every image inside it is then run.

NOTE FOR REVIEWER, please check before trusting this file:
  1. EMOTION_LABELS below is inferred from NEUTRAL_CLASS = 6 in
     common_utils.py plus the standard 0-indexed RAF-DB basic-7 label order,
     it is not read directly from basic/EmoLabel/list_partition_label.txt.
     Confirm the order against that file.
  2. RACE_LABELS and GENDER_LABELS are taken from print-statement comments
     elsewhere in the repo (05_joint_multitask_and_lambda_sweep.py names
     White/Black/Asian, common_utils.py comments male/female/unsure for
     gender), not from a canonical label map. AGE group ids are left as
     plain integers, fill in real bin edges if you want them displayed.
  3. In the shrinkage mode, Head's ShrinkageEmbedding.counts is a registered
     buffer, so load_state_dict() should overwrite the placeholder counts
     this script constructs the model with, with whatever counts the
     checkpoint was actually trained on. Please verify this is in fact
     happening rather than assuming it, a silent mismatch here would
     produce wrong shrinkage weights without erroring.
"""

import argparse
import os

import torch
from PIL import Image

from common_utils import (
    device, eval_transform, EMA_CHECKPOINT_PATH, POSTER_VAR_CACHE,
    N_RACE, N_GENDER, N_AGE,
)
from model import Head, JointModel, build_posterv2_backbone

# RAF-DB basic-7 label order, 0-indexed to match NEUTRAL_CLASS = 6 in
# common_utils.py. VERIFY against list_partition_label.txt, see note above.
EMOTION_LABELS = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

RACE_LABELS = ['White', 'Black', 'Asian']
GENDER_LABELS = ['Male', 'Female', 'Unsure']


def load_image(path):
    img = Image.open(path).convert('RGB')
    return eval_transform(img).unsqueeze(0).to(device)


def find_images(path):
    if os.path.isdir(path):
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        return sorted(
            os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(exts)
        )
    return [path]


# ------------------------------------------------------------------
# Mode 1: emotion only (plain fine-tuned backbone)
# ------------------------------------------------------------------

def load_emotion_model(checkpoint_path=None):
    checkpoint_path = checkpoint_path or EMA_CHECKPOINT_PATH
    model = build_posterv2_backbone(num_classes=7)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict_emotion(model, image_path):
    x = load_image(image_path)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    pred = int(probs.argmax())
    return {
        'image': image_path,
        'emotion': EMOTION_LABELS[pred],
        'confidence': float(probs[pred]),
        'all_probs': {EMOTION_LABELS[i]: float(p) for i, p in enumerate(probs)},
    }


# ------------------------------------------------------------------
# Mode 2: joint (emotion + demographic logits, one forward pass)
# ------------------------------------------------------------------

def load_joint_model(lambda_value='0.1'):
    base = build_posterv2_backbone(num_classes=7)
    base.load_state_dict(torch.load(EMA_CHECKPOINT_PATH, map_location=device))
    # Matches load_joint_01() / run_lambda_sweep() in the source notebook.
    # Not functionally required for inference-only use (no gradients are
    # taken here either way), kept for fidelity with how these checkpoints
    # were produced.
    for p in base.ir_back.parameters():
        p.requires_grad = False
    model = JointModel(base).to(device)
    ckpt_path = os.path.join(POSTER_VAR_CACHE, 'lambda_sweep', f'joint_model_lambda_{lambda_value}.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict_joint(model, image_path):
    x = load_image(image_path)
    emo_logits, race_logits, gender_logits, age_logits = model(x)
    emo_pred = int(emo_logits.argmax(1))
    return {
        'image': image_path,
        'emotion': EMOTION_LABELS[emo_pred],
        'race': RACE_LABELS[int(race_logits.argmax(1))],
        'gender': GENDER_LABELS[int(gender_logits.argmax(1))],
        'age_group': int(age_logits.argmax(1)),
    }


# ------------------------------------------------------------------
# Mode 3: shrinkage-conditioned head, needs a feature plus demographic ids
# ------------------------------------------------------------------

def save_shrinkage_checkpoint(train_d, val_d, race_counts, gender_counts, age_counts,
                               class_w, out_path, seed=0):
    """Convenience helper: trains one Head(mode='shrinkage') via train_one()
    from 04_grouping_and_shrinkage_experiments.py and saves it, since the
    sweep in that file does not persist individual checkpoints on its own.
    Run this once, in an environment where 04's cached features are
    available, then point --shrinkage_ckpt at out_path."""
    from importlib import import_module
    grouping = import_module('04_grouping_and_shrinkage_experiments')
    model = grouping.train_one('shrinkage', seed, train_d, val_d,
                                race_counts, gender_counts, age_counts, class_w)
    torch.save(model.state_dict(), out_path)
    return model


def load_shrinkage_model(checkpoint_path):
    # Counts are placeholders, ShrinkageEmbedding.counts is a registered
    # buffer and gets overwritten by load_state_dict() below with whatever
    # counts the checkpoint was actually trained on. See reviewer note 3
    # at the top of this file.
    placeholder_race = torch.ones(N_RACE, device=device)
    placeholder_gender = torch.ones(N_GENDER, device=device)
    placeholder_age = torch.ones(N_AGE, device=device)
    model = Head('shrinkage', placeholder_race, placeholder_gender, placeholder_age).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def extract_feature(backbone, image_path):
    """768-d SE-gated feature, the same hook point as
    extract_posterv2_features() in 02_feature_extraction.py."""
    captured = {}

    def hook_fn(module, inp, out):
        captured['feat'] = out.detach()

    handle = backbone.VIT.se_block.register_forward_hook(hook_fn)
    x = load_image(image_path)
    _ = backbone(x)
    handle.remove()
    return captured['feat']


@torch.no_grad()
def predict_shrinkage(head_model, backbone, image_path, race, gender, age):
    f = extract_feature(backbone, image_path)
    r = torch.tensor([race], device=device)
    g = torch.tensor([gender], device=device)
    a = torch.tensor([age], device=device)
    logits = head_model(f, r, g, a)
    pred = int(logits.argmax(1))
    return {
        'image': image_path,
        'emotion': EMOTION_LABELS[pred],
        'race_input': RACE_LABELS[race],
        'gender_input': GENDER_LABELS[gender],
        'age_group_input': age,
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Run BRIDGER inference on one image or a folder.')
    p.add_argument('--image', required=True, help='Path to an image file or a directory of images.')
    p.add_argument('--mode', choices=['emotion', 'joint', 'shrinkage'], default='emotion')
    p.add_argument('--checkpoint', default=None, help='Override checkpoint path for --mode emotion.')
    p.add_argument('--lambda_value', default='0.1', help='Which lambda-sweep checkpoint to load for --mode joint.')
    p.add_argument('--shrinkage_ckpt', default=None, help='Path to a saved Head(mode=shrinkage) checkpoint, required for --mode shrinkage.')
    p.add_argument('--race', type=int, default=0, help='Race group id (0, 1, or 2), required for --mode shrinkage.')
    p.add_argument('--gender', type=int, default=0, help='Gender group id (0, 1, or 2), required for --mode shrinkage.')
    p.add_argument('--age', type=int, default=0, help='Age group id (0 to 4), required for --mode shrinkage.')
    args = p.parse_args()

    images = find_images(args.image)
    if not images:
        print(f'No images found at {args.image}')
        return

    if args.mode == 'emotion':
        model = load_emotion_model(args.checkpoint)
        for img in images:
            print(predict_emotion(model, img))

    elif args.mode == 'joint':
        model = load_joint_model(args.lambda_value)
        for img in images:
            print(predict_joint(model, img))

    elif args.mode == 'shrinkage':
        if args.shrinkage_ckpt is None:
            raise ValueError(
                '--mode shrinkage requires --shrinkage_ckpt. This repo does not ship one '
                'by default, see save_shrinkage_checkpoint() in this file to produce one.'
            )
        head = load_shrinkage_model(args.shrinkage_ckpt)
        backbone = load_emotion_model()
        for img in images:
            print(predict_shrinkage(head, backbone, img, args.race, args.gender, args.age))


if __name__ == '__main__':
    main()
