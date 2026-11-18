"""Utilities for training lightweight heads on cached frozen-backbone features."""
import functools
import hashlib
import math
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)

from common import ROOT

FEATURES = ROOT / 'full' / 'features'
MANIFESTS = ROOT / 'full' / 'manifests'


def split_frame(dataset, part):
    return pd.read_csv(MANIFESTS / f'{dataset}_{part}.csv', dtype=str).reset_index(drop=True)


def classes_of(dataset):
    return list(_classes_of(dataset))


@functools.lru_cache(maxsize=None)
def _classes_of(dataset):
    frames = [split_frame(dataset, part) for part in ('train', 'val', 'test')]
    return tuple(sorted(set().union(*(set(frame['label']) for frame in frames))))


def fingerprint(model_name, spatial, frame):
    description = model_name + str(spatial) + frame[['path', 'label']].to_csv(index=False)
    return hashlib.sha256(description.encode()).hexdigest()[:24]


def load_cached(dataset, part, model_name='lcnet_100.ra2_in1k', spatial=True):
    frame = split_frame(dataset, part)
    key = fingerprint(model_name, spatial, frame)
    features = np.load(FEATURES / f'{key}.npy', mmap_mode='r')
    labels = np.load(FEATURES / f'{key}_labels.npy')
    lookup = {label: index for index, label in enumerate(classes_of(dataset))}
    expected = frame['label'].map(lookup).to_numpy()
    if not np.array_equal(expected, labels):
        raise RuntimeError(f'Cached labels do not match manifest for {dataset}/{part}')
    return torch.from_numpy(np.asarray(features, dtype=np.float32)), torch.from_numpy(labels.astype(np.int64))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CBAM(nn.Module):
    """Identical to the project CBAM (shared MLP r=32, 7x7 spatial conv)."""
    def __init__(self, channels, reduction=32, use_channel=True, use_spatial=True):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.use_channel, self.use_spatial = use_channel, use_spatial
        self.channel_mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(), nn.Conv2d(hidden, channels, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        if self.use_channel:
            x = x * torch.sigmoid(self.channel_mlp(x.mean((2, 3), keepdim=True)) + self.channel_mlp(x.amax((2, 3), keepdim=True)))
        if self.use_spatial:
            x = x * torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x


class SE(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, channels, 1))

    def forward(self, x):
        return x * torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)))


class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        y = self.conv(x.mean((2, 3)).unsqueeze(1)).squeeze(1)
        return x * torch.sigmoid(y)[:, :, None, None]


class Head(nn.Module):
    """attention -> GAP -> [optional frozen pretrained LCNet conv_head 512->1280 + hardswish] -> dropout -> linear."""
    def __init__(self, attention, channels, num_classes, dropout=0.3, conv_head=None):
        super().__init__()
        self.attention = attention if attention is not None else nn.Identity()
        self.conv_head = conv_head
        width = channels if conv_head is None else conv_head.out_channels
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(width, num_classes)

    def embed(self, x):
        x = self.attention(x).mean((2, 3), keepdim=True)
        if self.conv_head is not None:
            x = nn.functional.hardswish(self.conv_head(x))
        return x.flatten(1)

    def forward(self, x):
        return self.fc(self.dropout(self.embed(x)))


def pretrained_conv_head(path):
    state = torch.load(path, map_location='cpu', weights_only=True)['state']
    conv = nn.Conv2d(512, 1280, 1, bias=True)
    conv.weight.data.copy_(state['conv_head.weight'])
    conv.bias.data.copy_(state['conv_head.bias'])
    for parameter in conv.parameters():
        parameter.requires_grad_(False)
    return conv


def class_weights(labels, num_classes, beta=0.999):
    counts = np.bincount(labels.numpy(), minlength=num_classes).astype(np.float64)
    effective = (1 - np.power(beta, counts)) / (1 - beta)
    weights = 1 / effective
    return torch.tensor(weights / weights.sum() * num_classes, dtype=torch.float32)


def make_loss(kind, labels, num_classes):
    prior = torch.tensor(np.bincount(labels.numpy(), minlength=num_classes) / len(labels), dtype=torch.float32)
    if kind == 'ce':
        return lambda logits, y: nn.functional.cross_entropy(logits, y)
    if kind == 'wce':
        weights = class_weights(labels, num_classes)
        return lambda logits, y: nn.functional.cross_entropy(logits, y, weight=weights)
    if kind == 'ls':
        return lambda logits, y: nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
    if kind == 'focal':
        def focal(logits, y, gamma=2.0):
            logp = nn.functional.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
            return (-(1 - logp.exp()) ** gamma * logp).mean()
        return focal
    if kind == 'la':
        adjustment = prior.clamp_min(1e-12).log()
        return lambda logits, y: nn.functional.cross_entropy(logits + adjustment, y)
    raise ValueError(kind)


def predict(model, x, batch=512, mc=0):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = x[start:start + batch]
            if mc:
                model.dropout.train()
                outputs.append(torch.stack([model(chunk).softmax(1) for _ in range(mc)]))
                model.dropout.eval()
            else:
                outputs.append(model(chunk))
    return torch.cat(outputs, 1 if mc else 0)


def train(model, xtr, ytr, xva, yva, num_classes, loss='ce', epochs=30, patience=8, lr=3e-3, batch=64, seed=0, select='f1'):
    seed_all(seed)
    criterion = make_loss(loss, ytr, num_classes)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best, stale, state, history = -1.0, 0, None, []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(ytr), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            value = criterion(model(xtr[index]), ytr[index])
            value.backward()
            optimizer.step()
            total += float(value) * len(index)
        logits = predict(model, xva)
        predicted = logits.argmax(1).numpy()
        score = f1_score(yva.numpy(), predicted, average='macro', zero_division=0) if select == 'f1' else accuracy_score(yva.numpy(), predicted)
        history.append({'epoch': epoch + 1, 'loss': total / len(ytr), 'val_score': float(score)})
        if score > best:
            best, stale = score, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(state)
    return model, history


def ece_score(probabilities, labels, bins=15):
    confidence = probabilities.max(1)
    correct = (probabilities.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def full_metrics(probabilities, labels, num_classes, ordinal=False):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    predicted = probabilities.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predicted, labels=list(range(num_classes)), zero_division=0)
    matrix = confusion_matrix(labels, predicted, labels=list(range(num_classes)))
    specificity = []
    for k in range(num_classes):
        tn = matrix.sum() - matrix[k].sum() - matrix[:, k].sum() + matrix[k, k]
        fp = matrix[:, k].sum() - matrix[k, k]
        specificity.append(tn / (tn + fp) if tn + fp else 0.0)
    onehot = np.eye(num_classes)[labels]
    result = {
        'accuracy': float(accuracy_score(labels, predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predicted)),
        'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()), 'macro_f1': float(f1.mean()),
        'weighted_f1': float(f1_score(labels, predicted, average='weighted', zero_division=0)),
        'macro_specificity': float(np.mean(specificity)),
        'macro_auc_ovr': float(roc_auc_score(labels, probabilities, multi_class='ovr', average='macro', labels=list(range(num_classes)))),
        'ece': ece_score(probabilities, labels),
        'brier': float(((probabilities - onehot) ** 2).sum(1).mean()),
        'nll': float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
        'per_class': {'precision': precision.tolist(), 'recall': recall.tolist(), 'f1': f1.tolist(), 'specificity': [float(s) for s in specificity], 'support': support.tolist(),
                      'auc': [float(roc_auc_score(onehot[:, k], probabilities[:, k])) for k in range(num_classes)]},
        'confusion': matrix.tolist(),
    }
    if ordinal:
        result['qwk'] = float(cohen_kappa_score(labels, predicted, weights='quadratic'))
        result['mae_grades'] = float(np.abs(labels - predicted).mean())
    return result


def bootstrap_ci(labels, predicted, metric, rounds=1000, seed=0):
    rng = np.random.default_rng(seed)
    values = []
    n = len(labels)
    for _ in range(rounds):
        index = rng.integers(0, n, n)
        values.append(metric(labels[index], predicted[index]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
