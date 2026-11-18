"""Experiment 2: attention / head / loss ablation on frozen LCNet-1.0 spatial features.

Disease task (HAM10000, 7 classes) and severity task (ACNE04, 4 grades). Five seeds per
configuration; model selection on validation macro-F1 unless stated otherwise.
Saves per-run test probabilities (deterministic and 30-pass MC dropout) for later analysis.
"""
import json
import sys
import time

import numpy as np
import torch

from common import RESULTS, ROOT
from heads import (CBAM, ECA, SE, Head, classes_of, full_metrics, load_cached, predict, pretrained_conv_head, train)

torch.set_num_threads(4)
SEEDS = [0, 1, 2, 3, 4]
OUT = RESULTS / 'exp2'
OUT.mkdir(parents=True, exist_ok=True)
LCNET_CKPT = ROOT / 'full' / 'HAM10000_lcnet_100.ra2_in1k.pt'


def attention(kind, channels=512):
    return {'none': None, 'cbam': CBAM(channels), 'cbam_channel': CBAM(channels, use_spatial=False), 'cbam_spatial': CBAM(channels, use_channel=False),
            'se': SE(channels), 'eca': ECA(channels)}[kind]


def build(att, convhead, num_classes):
    return Head(attention(att), 512, num_classes, 0.3, pretrained_conv_head(LCNET_CKPT) if convhead else None)


def run(task, dataset, att, convhead, loss, select, epochs, seeds, data, frozen_attention=None, tag=None, patience=8):
    (xtr, ytr), (xva, yva), (xte, yte) = data
    num_classes = len(classes_of(dataset))
    name = tag or f'{task}__{att}{"+convhead" if convhead else ""}__{loss}__sel-{select}__ep{epochs}'
    path = OUT / f'{name}.json'
    if path.exists():
        return json.loads(path.read_text())
    runs = []
    for seed in seeds:
        start = time.time()
        torch.manual_seed(seed)
        model = build(att, convhead, num_classes)
        if frozen_attention is not None:
            model.attention.load_state_dict(frozen_attention)
            for parameter in model.attention.parameters():
                parameter.requires_grad_(False)
        model, history = train(model, xtr, ytr, xva, yva, num_classes, loss=loss, epochs=epochs, patience=patience, seed=seed, select=select)
        probs = predict(model, xte).softmax(1).numpy()
        mc = predict(model, xte, mc=30).numpy()
        val_probs = predict(model, xva).softmax(1).numpy()
        np.savez_compressed(OUT / f'{name}__seed{seed}.npz', test=probs, test_mc=mc.astype(np.float16), val=val_probs, ytest=yte.numpy(), yval=yva.numpy())
        metrics = full_metrics(probs, yte.numpy(), num_classes, ordinal=task == 'severity')
        metrics['mc'] = full_metrics(mc.mean(0), yte.numpy(), num_classes, ordinal=task == 'severity')
        metrics['epochs_run'] = len(history)
        metrics['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics['head_params_total'] = sum(p.numel() for p in model.parameters())
        metrics['seconds'] = time.time() - start
        runs.append(metrics)
        print(f'{name} seed {seed}: acc {metrics["accuracy"]:.4f} macroF1 {metrics["macro_f1"]:.4f} bal {metrics["balanced_accuracy"]:.4f} ({metrics["seconds"]:.0f}s)', flush=True)
    keys = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'macro_auc_ovr', 'ece', 'brier', 'nll'] + (['qwk', 'mae_grades'] if task == 'severity' else [])
    summary = {key: [float(np.mean([r[key] for r in runs])), float(np.std([r[key] for r in runs]))] for key in keys}
    summary_mc = {key: [float(np.mean([r['mc'][key] for r in runs])), float(np.std([r['mc'][key] for r in runs]))] for key in keys}
    result = {'name': name, 'task': task, 'attention': att, 'convhead': convhead, 'loss': loss, 'select': select, 'epochs': epochs, 'patience': patience, 'n_seeds': len(seeds), 'summary': summary, 'summary_mc': summary_mc, 'runs': runs}
    path.write_text(json.dumps(result, indent=1))
    return result


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('disease', 'all'):
        data = [load_cached('HAM10000', part) for part in ('train', 'val', 'test')]
        # Faithful re-run of the project's protocol (CBAM, CE, 10 epochs, accuracy selection)
        run('disease', 'HAM10000', 'cbam', False, 'ce', 'acc', 10, SEEDS, data)
        for att in ('none', 'cbam', 'cbam_channel', 'cbam_spatial', 'se', 'eca'):
            run('disease', 'HAM10000', att, False, 'ce', 'f1', 30, SEEDS, data)
        for att in ('none', 'cbam'):
            run('disease', 'HAM10000', att, True, 'ce', 'f1', 30, SEEDS, data)
        for att, convhead in (('none', True), ('cbam', False)):
            for loss in ('wce', 'la', 'focal', 'ls'):
                run('disease', 'HAM10000', att, convhead, loss, 'f1', 30, SEEDS, data)
        run('disease', 'HAM10000', 'cbam', True, 'wce', 'f1', 30, SEEDS, data)
        run('disease', 'HAM10000', 'cbam', True, 'la', 'f1', 30, SEEDS, data)
    if stage in ('severity', 'all'):
        data = [load_cached('ACNE04', part) for part in ('train', 'val', 'test')]
        dual = torch.load(ROOT / 'full' / 'dual_disease.pt', map_location='cpu', weights_only=True)['state']
        shared_cbam = {k.replace('cbam.', ''): v for k, v in dual.items() if k.startswith('cbam.')}
        run('severity', 'ACNE04', 'cbam', False, 'ce', 'acc', 15, SEEDS, data, frozen_attention=shared_cbam, tag='severity__shared-HAM-cbam(frozen)__ce__sel-acc__ep15')
        run('severity', 'ACNE04', 'cbam', False, 'ce', 'f1', 30, SEEDS, data, frozen_attention=shared_cbam, tag='severity__shared-HAM-cbam(frozen)__ce__sel-f1__ep30')
        for att, convhead in (('none', False), ('cbam', False), ('none', True), ('cbam', True)):
            for loss in ('ce', 'wce'):
                run('severity', 'ACNE04', att, convhead, loss, 'f1', 30, SEEDS, data)
    print('EXP2 DONE', stage)

