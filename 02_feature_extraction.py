"""
02_feature_extraction.py

  1. Extract the 768-d SE-gated POSTER-Var feature (hooked at VIT.se_block)
     for train/val/test and cache to Drive.
  2. Extract pristine IR50 (face-recognition, pre-expression-training)
     features for train/test.
  3. Extract the IR50 stream from *inside* the fine-tuned POSTER-Var model
     (post-expression-training) for train/test.

These three feature sets are what `03_decodability_analysis.py` compares.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common_utils import (
    RAFDB_IMG_DIR, CSV_DIR, POSTER_VAR_DIR, EMA_CHECKPOINT_PATH,
    FEATURE_CACHE, device, eval_transform, RafDbDataset, load_pyramid_model,
)


# ============================================================
# 1. POSTER-Var 768-d feature extraction (EMA checkpoint)
# ============================================================
def extract_posterv2_features():
    pyramid_trans_expr2 = load_pyramid_model()
    model = pyramid_trans_expr2(img_size=224, num_classes=7, vae=True).to(device)
    model.load_state_dict(torch.load(EMA_CHECKPOINT_PATH, map_location=device))
    model.eval()

    captured = {}

    def hook_fn(module, inp, out):
        captured['feat'] = out.detach()

    hook_handle = model.VIT.se_block.register_forward_hook(hook_fn)

    train_dataset_plain = RafDbDataset(os.path.join(CSV_DIR, 'train_labels.csv'), RAFDB_IMG_DIR, eval_transform)
    val_dataset = RafDbDataset(os.path.join(CSV_DIR, 'valid_labels.csv'), RAFDB_IMG_DIR, eval_transform)
    test_dataset = RafDbDataset(os.path.join(CSV_DIR, 'test_labels.csv'), RAFDB_IMG_DIR, eval_transform)

    extract_loaders = {
        'train': DataLoader(train_dataset_plain, batch_size=64, shuffle=False, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2),
        'test': DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2),
    }

    os.makedirs(FEATURE_CACHE, exist_ok=True)
    for split, loader in extract_loaders.items():
        feats, labels_all = [], []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                _ = model(inputs)  # forward pass triggers the hook
                feats.append(captured['feat'].cpu())
                labels_all.append(labels)
        feats = torch.cat(feats, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        torch.save({'features': feats, 'labels': labels_all}, FEATURE_CACHE + f'posterv2_ema_{split}.pt')
        print(f"{split}: features {tuple(feats.shape)}, labels {tuple(labels_all.shape)}")

    hook_handle.remove()
    print("\nFeature extraction done. Cached to:", FEATURE_CACHE)


# ============================================================
# 2. Pristine IR50 features (no expression training applied)
# ============================================================
def extract_pristine_ir50():
    from trails.posterv2.ir50 import Backbone
    from trails.posterv2.PosterV2_7cls import load_pretrained_weights

    ir50 = Backbone(50, 0.0, 'ir').to(device)
    ir50_ckpt = torch.load(f'{POSTER_VAR_DIR}models/pretrain/ir50.pth', map_location=device)
    ir50 = load_pretrained_weights(ir50, ir50_ckpt)
    ir50.eval()

    ir_feat = {}

    def ir_hook(module, inp, out):
        feats = out[-1] if isinstance(out, (tuple, list)) else out
        ir_feat['f'] = F.adaptive_avg_pool2d(feats, 1).flatten(1).detach()

    hook_handle = ir50.register_forward_hook(ir_hook)

    def extract_pristine_ir(csv_name):
        ds = RafDbDataset(os.path.join(CSV_DIR, csv_name), RAFDB_IMG_DIR, eval_transform)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        feats = []
        with torch.no_grad():
            for imgs, _ in loader:
                _ = ir50(imgs.to(device))
                feats.append(ir_feat['f'].cpu())
        return torch.cat(feats, dim=0)

    print("Extracting pristine IR50 features (runs the backbone, a few minutes)...")
    ir_train = extract_pristine_ir('train_labels.csv').to(device)
    ir_test = extract_pristine_ir('test_labels.csv').to(device)
    hook_handle.remove()
    print("Pristine IR50 feature dim:", ir_train.shape[1])
    return ir_train, ir_test


# ============================================================
# 3. IR50 stream inside the fine-tuned POSTER-Var (post-expression)
# ============================================================
def extract_ir_stream_inside_posterv2():
    pyramid_trans_expr2 = load_pyramid_model()
    pv = pyramid_trans_expr2(img_size=224, num_classes=7, vae=True).to(device)
    pv.load_state_dict(torch.load(EMA_CHECKPOINT_PATH, map_location=device))
    pv.eval()

    tapped = {}

    def ir_stream_hook(module, inp, out):
        feats = out[-1] if isinstance(out, (tuple, list)) else out
        tapped['ir_stream'] = F.adaptive_avg_pool2d(feats, 1).flatten(1).detach()

    h = pv.ir_back.register_forward_hook(ir_stream_hook)

    def extract_ir_stream(csv_name):
        ds = RafDbDataset(os.path.join(CSV_DIR, csv_name), RAFDB_IMG_DIR, eval_transform)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        feats = []
        with torch.no_grad():
            for imgs, _ in loader:
                _ = pv(imgs.to(device))
                feats.append(tapped['ir_stream'].cpu())
        return torch.cat(feats, dim=0)

    print("Extracting fine-tuned IR50 stream features from inside POSTER-Var...")
    irstream_train = extract_ir_stream('train_labels.csv').to(device)
    irstream_test = extract_ir_stream('test_labels.csv').to(device)
    h.remove()
    print("Fine-tuned IR50 stream dim:", irstream_train.shape[1])
    return irstream_train, irstream_test


# ============================================================
# Alternate: dynamically locate the IR50 Backbone module by class name
# instead of assuming the `ir_back` attribute (used as a diagnostic in the
# notebook when the attribute path was uncertain).
# ============================================================
def extract_ir_stream_dynamic_lookup():
    pyramid_trans_expr2 = load_pyramid_model()
    model_diagnostic = pyramid_trans_expr2(img_size=224, num_classes=7, vae=True).to(device)
    model_diagnostic.load_state_dict(torch.load(EMA_CHECKPOINT_PATH, map_location=device))
    model_diagnostic.eval()

    target_module = None
    for name, module in model_diagnostic.named_modules():
        if module.__class__.__name__ == 'Backbone':
            target_module = module
            print(f"Found IR50 backbone at: '{name}'")
            break

    if target_module is None:
        print("Could not find Backbone module. Here are the top-level modules:")
        for name, module in model_diagnostic.named_children():
            print(name, module.__class__.__name__)
        return None, None

    captured_ir = {}

    def poster_ir_hook(module, inp, out):
        feats = out[-1] if isinstance(out, (tuple, list)) else out
        captured_ir['f'] = F.adaptive_avg_pool2d(feats, 1).flatten(1).detach()

    hook_handle = target_module.register_forward_hook(poster_ir_hook)

    def extract_poster_ir(csv_name):
        ds = RafDbDataset(os.path.join(CSV_DIR, csv_name), RAFDB_IMG_DIR, eval_transform)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        feats = []
        with torch.no_grad():
            for imgs, _ in loader:
                _ = model_diagnostic(imgs.to(device))
                feats.append(captured_ir['f'].cpu())
        return torch.cat(feats, dim=0)

    print("Extracting POSTER-Var fine-tuned IR50 features...")
    poster_ir_train = extract_poster_ir('train_labels.csv')
    poster_ir_test = extract_poster_ir('test_labels.csv')
    hook_handle.remove()
    print("POSTER-Var fine-tuned IR50 feature dim:", poster_ir_train.shape[1])
    return poster_ir_train, poster_ir_test


if __name__ == '__main__':
    extract_posterv2_features()
    extract_pristine_ir50()
    extract_ir_stream_inside_posterv2()
