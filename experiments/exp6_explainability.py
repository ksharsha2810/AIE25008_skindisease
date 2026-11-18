"""Experiment 6: quantitative explanation analysis on HAM10000 test images using the public
HAM10000 lesion segmentation masks (Tschandl et al., Harvard Dataverse doi:10.7910/DVN/DBW86T).

Methods: Grad-CAM, Grad-CAM++, Eigen-CAM (gradient-free) at the last LCNet block (512x7x7).
Metrics per image (predicted-class target):
  energy_in_lesion   fraction of CAM mass inside the lesion mask
  energy_ratio       energy_in_lesion / lesion-area fraction (1.0 = uniform map)
  pointing_hit       CAM argmax falls inside the lesion
  centre_prior       same metrics for a fixed centred Gaussian (trivial baseline)
  deletion drop      drop of target probability after replacing the top-20% CAM pixels by a blurred copy,
                     compared with deleting a random 20% of pixels (faithfulness sanity check)
Also correlates explanation localisation with MC-dropout uncertainty and correctness.
Usage: python exp6_explainability.py [attention,convhead|plain,loss] [max_images]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.stats import mannwhitneyu, spearmanr
from torchvision import transforms

from common import PROJECT, RESULTS
from heads import classes_of
from models import Full, backbone, geometry, heads_under_test, normalizer, parse_proposed

torch.set_num_threads(2)
MASKS = PROJECT / 'skin_pipeline_artifacts' / 'data' / 'HAM10000' / 'masks'
_INDEX = {}


def mask_for(path):
    if not _INDEX:
        _INDEX.update({p.stem.replace('_segmentation', ''): p for p in MASKS.rglob('*_segmentation.png')})
    return _INDEX.get(Path(path).stem)


def centre_prior(size=224, sigma=0.25):
    grid = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(grid, grid)
    return np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))


def localisation(cam, mask):
    cam = np.clip(cam, 0, None)
    total = cam.sum()
    energy = float((cam * mask).sum() / total) if total > 0 else float('nan')
    area = float(mask.mean())
    peak = np.unravel_index(int(cam.argmax()), cam.shape)
    return energy, energy / area if area > 0 else float('nan'), bool(mask[peak] > 0), area


def norm_entropy(p):
    return float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(len(p)))


if __name__ == '__main__':
    proposed = parse_proposed(sys.argv)
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    frame = pd.read_csv(RESULTS.parent / 'root' / 'full' / 'manifests' / 'HAM10000_test.csv', dtype=str)
    lookup = {c: i for i, c in enumerate(classes_of('HAM10000'))}
    frame = frame.assign(mask=frame['path'].map(mask_for))
    print('masks found for', frame['mask'].notna().sum(), 'of', len(frame), 'test images', flush=True)
    frame = frame[frame['mask'].notna()].reset_index(drop=True)
    if limit:
        frame = frame.sample(n=min(limit, len(frame)), random_state=0).reset_index(drop=True)
    crop, cfg = geometry()
    mask_crop = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST), transforms.CenterCrop(224)])
    norm = normalizer()
    trunk = backbone()
    prior = centre_prior()
    rng = np.random.default_rng(0)
    rows = []
    examples = {}
    for head_name, head in heads_under_test(proposed).items():
        model = Full(trunk, head).eval()
        layer = trunk.blocks[-1]
        explainers = {'Grad-CAM': GradCAM(model=model, target_layers=[layer]), 'Grad-CAM++': GradCAMPlusPlus(model=model, target_layers=[layer]), 'Eigen-CAM': EigenCAM(model=model, target_layers=[layer])}
        for index, item in frame.iterrows():
            with Image.open(item['path']) as image:
                rgb = crop(image.convert('RGB'))
            with Image.open(item['mask']) as m:
                mask = (np.asarray(mask_crop(m.convert('L'))) > 127).astype(np.float32)
            x = norm(rgb).unsqueeze(0)
            with torch.no_grad():
                features = trunk.forward_features(x)
                prob = head(features).softmax(1)[0].numpy()
                torch.manual_seed(42)
                head.dropout.train()
                mc = torch.stack([head(features).softmax(1)[0] for _ in range(30)]).mean(0).numpy()
                head.dropout.eval()
            target = int(prob.argmax())
            blurred = norm(rgb.filter(ImageFilter.GaussianBlur(10))).unsqueeze(0)
            row = {'head': head_name, 'image': Path(item['path']).stem, 'label': lookup[item['label']], 'pred': target, 'correct': target == lookup[item['label']],
                   'confidence': float(prob.max()), 'mc_entropy': norm_entropy(mc)}
            e, r, hit, area = localisation(prior, mask)
            row.update({'lesion_area': area, 'centre_prior_energy': e, 'centre_prior_ratio': r, 'centre_prior_hit': hit})
            random_keep = torch.from_numpy((rng.random((224, 224)) >= 0.2).astype(np.float32))
            with torch.no_grad():
                random_drop = float(prob[target] - model(x * random_keep + blurred * (1 - random_keep)).softmax(1)[0, target])
            row['random20_deletion_drop'] = random_drop
            for method, explainer in explainers.items():
                cam = explainer(input_tensor=x, targets=[ClassifierOutputTarget(target)])[0]
                e, r, hit, _ = localisation(cam, mask)
                keep = torch.from_numpy((cam < np.quantile(cam, 0.8)).astype(np.float32))
                with torch.no_grad():
                    drop = float(prob[target] - model(x * keep + blurred * (1 - keep)).softmax(1)[0, target])
                row.update({f'{method}_energy': e, f'{method}_ratio': r, f'{method}_hit': hit, f'{method}_deletion_drop': drop})
                if index < 6 and method == 'Grad-CAM':
                    examples.setdefault(head_name, []).append({'image': row['image'], 'cam': cam.astype(np.float16), 'mask': mask.astype(np.uint8), 'rgb': np.asarray(rgb), 'label': row['label'], 'pred': target, 'conf': row['confidence'], 'entropy': row['mc_entropy']})
            rows.append(row)
            if index % 200 == 0:
                print(head_name, index, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / 'exp6_explainability_per_image.csv', index=False)
    summary = {}
    for head_name, g in df.groupby('head'):
        s = {'n': len(g), 'lesion_area_mean': float(g['lesion_area'].mean()), 'centre_prior': {'energy': float(g['centre_prior_energy'].mean()), 'ratio': float(g['centre_prior_ratio'].mean()), 'hit': float(g['centre_prior_hit'].mean())},
             'random20_deletion_drop': float(g['random20_deletion_drop'].mean())}
        for method in ('Grad-CAM', 'Grad-CAM++', 'Eigen-CAM'):
            s[method] = {'energy': float(g[f'{method}_energy'].mean()), 'ratio': float(g[f'{method}_ratio'].mean()), 'hit': float(g[f'{method}_hit'].mean()), 'deletion_drop_top20': float(g[f'{method}_deletion_drop'].mean()),
                         'energy_correct': float(g.loc[g.correct, f'{method}_energy'].mean()), 'energy_wrong': float(g.loc[~g.correct, f'{method}_energy'].mean()),
                         'energy_correct_vs_wrong_mwu_p': float(mannwhitneyu(g.loc[g.correct, f'{method}_energy'], g.loc[~g.correct, f'{method}_energy']).pvalue),
                         'spearman_energy_vs_entropy': [float(v) for v in spearmanr(g[f'{method}_energy'], g['mc_entropy'])]}
        summary[head_name] = s
        print(head_name, json.dumps(s, indent=1))
    (RESULTS / 'exp6_explainability.json').write_text(json.dumps(summary, indent=1))
    np.save(RESULTS / 'exp6_examples.npy', examples, allow_pickle=True)
    print('EXP6 DONE')

