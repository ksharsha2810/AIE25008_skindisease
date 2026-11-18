"""Experiment 5: robustness to acquisition degradations (HAM10000 test set, 2,003 images).

Corruptions are applied after the deterministic resize/centre-crop, before normalisation.
For each head: accuracy, macro-F1, ECE, mean MC-dropout normalized entropy, fraction flagged
by the 0.4 review rule, and misclassification AUROC of the entropy score.
Usage: python exp5_robustness.py [attention,convhead|plain,loss]
"""
import io
import json
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import f1_score, roc_auc_score

from common import RESULTS
from heads import classes_of, ece_score
from models import backbone, geometry, heads_under_test, normalizer, parse_proposed

torch.set_num_threads(2)
LADDERS = {
    'gaussian_noise': [0.02, 0.04, 0.06, 0.08, 0.10], 'gaussian_blur': [0.5, 1.0, 1.5, 2.0, 2.5], 'darken': [0.9, 0.8, 0.7, 0.6, 0.5], 'brighten': [1.1, 1.2, 1.3, 1.4, 1.5],
    'contrast': [0.9, 0.8, 0.7, 0.6, 0.5], 'jpeg': [80, 60, 40, 20, 10], 'low_resolution': [160, 128, 96, 64, 32],
}
EVALUATED_SEVERITIES = (1, 3, 5)
N_IMAGES = 300
_UNUSED = {
    'gaussian_noise': [0.04, 0.08, 0.12], 'gaussian_blur': [1, 2, 3], 'darken': [0.8, 0.6, 0.4], 'brighten': [1.2, 1.4, 1.6],
    'contrast': [0.8, 0.6, 0.4], 'jpeg': [50, 25, 10], 'low_resolution': [112, 64, 32],
}


def corrupt(image, kind, level, rng):
    if kind == 'clean':
        return image
    if kind == 'gaussian_noise':
        array = np.asarray(image, dtype=np.float32) / 255 + rng.normal(0, level, (224, 224, 3))
        return Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
    if kind == 'gaussian_blur':
        return image.filter(ImageFilter.GaussianBlur(level))
    if kind in ('darken', 'brighten'):
        return ImageEnhance.Brightness(image).enhance(level)
    if kind == 'contrast':
        return ImageEnhance.Contrast(image).enhance(level)
    if kind == 'jpeg':
        buffer = io.BytesIO()
        image.save(buffer, 'JPEG', quality=level)
        return Image.open(io.BytesIO(buffer.getvalue())).convert('RGB')
    if kind == 'low_resolution':
        return image.resize((level, level), Image.BICUBIC).resize((224, 224), Image.BICUBIC)
    raise ValueError(kind)


def norm_entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(-1) / np.log(p.shape[-1])


if __name__ == '__main__':
    proposed = parse_proposed(sys.argv)
    frame = pd.read_csv(RESULTS.parent / 'root' / 'full' / 'manifests' / 'HAM10000_test.csv', dtype=str)
    per_class = (frame['label'].value_counts() / len(frame) * N_IMAGES).round().astype(int).clip(lower=10)
    frame = pd.concat([g.sample(n=min(len(g), int(per_class[c])), random_state=0) for c, g in frame.groupby('label')]).reset_index(drop=True)
    print('robustness set', len(frame), frame['label'].value_counts().to_dict(), flush=True)
    lookup = {c: i for i, c in enumerate(classes_of('HAM10000'))}
    labels = frame['label'].map(lookup).to_numpy()
    crop, _ = geometry()
    norm = normalizer()
    images = []
    for path in frame['path']:
        with Image.open(path) as image:
            images.append(crop(image.convert('RGB')))
    trunk = backbone()
    heads = heads_under_test(proposed)
    conditions = [('clean', 0, 0)] + [(k, levels[s - 1], s) for k, levels in LADDERS.items() for s in EVALUATED_SEVERITIES]
    rows = []
    for kind, level, severity in conditions:
        rng = np.random.default_rng(1234)
        features = []
        with torch.no_grad():
            for start in range(0, len(images), 64):
                batch = torch.stack([norm(corrupt(img, kind, level, rng)) for img in images[start:start + 64]])
                features.append(trunk.forward_features(batch))
        features = torch.cat(features)
        for name, head in heads.items():
            with torch.no_grad():
                det = torch.cat([head(features[i:i + 256]).softmax(1) for i in range(0, len(features), 256)]).numpy()
                torch.manual_seed(42)
                head.dropout.train()
                mc = np.stack([torch.cat([head(features[i:i + 256]).softmax(1) for i in range(0, len(features), 256)]).numpy() for _ in range(30)]).mean(0)
                head.dropout.eval()
            entropy = norm_entropy(mc)
            predicted = mc.argmax(1)
            wrong = (predicted != labels).astype(int)
            rows.append({'head': name, 'corruption': kind, 'level': level, 'severity': severity,
                         'accuracy': float((det.argmax(1) == labels).mean()), 'macro_f1': float(f1_score(labels, det.argmax(1), average='macro')),
                         'ece': ece_score(det, labels), 'mc_accuracy': float((predicted == labels).mean()), 'mean_entropy': float(entropy.mean()),
                         'flag_rate_0.4': float((entropy > 0.4).mean()), 'misclf_auroc_entropy': float(roc_auc_score(wrong, entropy)),
                         'nevus_prediction_rate': float((det.argmax(1) == lookup['nevus']).mean())})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / 'exp5_robustness.csv', index=False)
    print('EXP5 DONE')

