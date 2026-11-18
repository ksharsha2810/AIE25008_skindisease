"""Experiment 3: label-compatible cross-dataset evaluation between HAM10000 (dermoscopic)
and PAD-UFES-20 (smartphone clinical images), using frozen LCNet-1.0 pooled features.

Only diagnostically compatible classes are used:
  strict (3): BCC, MEL, NEV  -- identical diagnostic definitions in both datasets
  extended (5): + AK (HAM 'akiec' includes Bowen's disease; PAD 'ack' does not)
                + BKL/SEK (HAM 'bkl' is a superset of PAD 'sek')   -> reported as approximate
"""
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from common import RESULTS
from heads import Head, classes_of, full_metrics, load_cached, predict, train

torch.set_num_threads(4)
SEEDS = [0, 1, 2, 3, 4]
MAPS = {
    'strict': {'HAM10000': {'basal_cell_carcinoma': 'BCC', 'melanoma': 'MEL', 'nevus': 'NEV'},
               'PAD_UFES_20': {'basal_cell_carcinoma': 'BCC', 'melanoma': 'MEL', 'nevus': 'NEV'}},
    'extended': {'HAM10000': {'basal_cell_carcinoma': 'BCC', 'melanoma': 'MEL', 'nevus': 'NEV', 'actinic_keratosis_or_intraepithelial_carcinoma': 'AK', 'benign_keratosis': 'BKL/SEK'},
                 'PAD_UFES_20': {'basal_cell_carcinoma': 'BCC', 'melanoma': 'MEL', 'nevus': 'NEV', 'actinic_keratosis': 'AK', 'seborrheic_keratosis': 'BKL/SEK'}},
}


def pooled(dataset, part):
    x, y = load_cached(dataset, part, spatial=False)
    return x[:, :, None, None], y


def restrict(x, y, dataset, mapping, shared):
    names = classes_of(dataset)
    keep = np.array([names[i] in mapping for i in y.numpy()])
    new = torch.tensor([shared.index(mapping[names[i]]) for i in y.numpy()[keep]])
    return x[torch.from_numpy(keep)], new


def fit_eval(train_sets, val_sets, test_sets, k, seed, init=None, epochs=30):
    xtr = torch.cat([s[0] for s in train_sets]); ytr = torch.cat([s[1] for s in train_sets])
    xva = torch.cat([s[0] for s in val_sets]); yva = torch.cat([s[1] for s in val_sets])
    torch.manual_seed(seed)
    model = Head(None, 1280, k, 0.3)
    if init is not None:
        model.load_state_dict(init)
    model, _ = train(model, xtr, ytr, xva, yva, k, loss='wce', epochs=epochs, patience=8, seed=seed, select='f1')
    out = {}
    for name, (xte, yte) in test_sets.items():
        probs = predict(model, xte).softmax(1).numpy()
        out[name] = full_metrics(probs, yte.numpy(), k)
        out[name]['_probs'] = probs
    return model, out


results = {}
data = {d: {p: pooled(d, p) for p in ('train', 'val', 'test')} for d in ('HAM10000', 'PAD_UFES_20')}
for variant, maps in MAPS.items():
    shared = sorted(set(maps['HAM10000'].values()))
    k = len(shared)
    r = {d: {p: restrict(*data[d][p], d, maps[d], shared) for p in data[d]} for d in data}
    pad_all = (torch.cat([r['PAD_UFES_20'][p][0] for p in ('train', 'val', 'test')]), torch.cat([r['PAD_UFES_20'][p][1] for p in ('train', 'val', 'test')]))
    ham_all = (torch.cat([r['HAM10000'][p][0] for p in ('train', 'val', 'test')]), torch.cat([r['HAM10000'][p][1] for p in ('train', 'val', 'test')]))
    protocols = {}
    for seed in SEEDS:
        tests = {'HAM_test': r['HAM10000']['test'], 'PAD_test': r['PAD_UFES_20']['test'], 'PAD_all': pad_all, 'HAM_all': ham_all}
        ham_model, ham_out = fit_eval([r['HAM10000']['train']], [r['HAM10000']['val']], tests, k, seed)
        _, pad_out = fit_eval([r['PAD_UFES_20']['train']], [r['PAD_UFES_20']['val']], tests, k, seed)
        _, joint_out = fit_eval([r['HAM10000']['train'], r['PAD_UFES_20']['train']], [r['HAM10000']['val'], r['PAD_UFES_20']['val']], tests, k, seed)
        _, ft_out = fit_eval([r['PAD_UFES_20']['train']], [r['PAD_UFES_20']['val']], tests, k, seed, init=ham_model.state_dict())
        rows = {
            'HAM->HAM (in-domain)': ham_out['HAM_test'], 'HAM->PAD test (zero-shot)': ham_out['PAD_test'], 'HAM->PAD all (zero-shot)': ham_out['PAD_all'],
            'PAD->PAD (in-domain)': pad_out['PAD_test'], 'PAD->HAM test (zero-shot)': pad_out['HAM_test'], 'PAD->HAM all (zero-shot)': pad_out['HAM_all'],
            'HAM+PAD joint -> HAM test': joint_out['HAM_test'], 'HAM+PAD joint -> PAD test': joint_out['PAD_test'],
            'HAM init, PAD fine-tune -> PAD test': ft_out['PAD_test'], 'HAM init, PAD fine-tune -> HAM test': ft_out['HAM_test'],
        }
        for name, metrics in rows.items():
            protocols.setdefault(name, []).append({key: metrics[key] for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'macro_auc_ovr', 'ece')} | {'recall': metrics['per_class']['recall'], 'n': int(np.sum(metrics['per_class']['support']))})
    summary = {}
    for name, runs in protocols.items():
        summary[name] = {key: [float(np.mean([x[key] for x in runs])), float(np.std([x[key] for x in runs]))] for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'macro_auc_ovr', 'ece')}
        summary[name]['per_class_recall_mean'] = np.mean([x['recall'] for x in runs], 0).tolist()
        summary[name]['n'] = runs[0]['n']
        print(variant, name, {k2: round(v[0], 4) for k2, v in summary[name].items() if isinstance(v, list) and len(v) == 2})
    results[variant] = {'classes': shared, 'train_counts': {d: np.bincount(r[d]['train'][1].numpy(), minlength=k).tolist() for d in r}, 'test_counts': {d: np.bincount(r[d]['test'][1].numpy(), minlength=k).tolist() for d in r}, 'summary': summary}

# Uncertainty under acquisition shift: full 7-class HAM linear probe (project checkpoint) applied to PAD images.
state = torch.load(RESULTS.parent / 'root' / 'full' / 'HAM10000_lcnet_100.ra2_in1k.pt', map_location='cpu', weights_only=True)['state']
W, b = state['classifier.weight'], state['classifier.bias']
def entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(1) / np.log(p.shape[1])
ham_p = torch.softmax(data['HAM10000']['test'][0].flatten(1) @ W.T + b, 1).numpy()
pad_x = torch.cat([data['PAD_UFES_20'][p][0] for p in ('train', 'val', 'test')]).flatten(1)
pad_p = torch.softmax(pad_x @ W.T + b, 1).numpy()
scores = np.concatenate([entropy(ham_p), entropy(pad_p)]); target = np.concatenate([np.zeros(len(ham_p)), np.ones(len(pad_p))])
msp = np.concatenate([1 - ham_p.max(1), 1 - pad_p.max(1)])
results['shift_uncertainty'] = {
    'ham_test_mean_norm_entropy': float(entropy(ham_p).mean()), 'pad_all_mean_norm_entropy': float(entropy(pad_p).mean()),
    'ham_test_frac_flagged_0.4': float((entropy(ham_p) > 0.4).mean()), 'pad_all_frac_flagged_0.4': float((entropy(pad_p) > 0.4).mean()),
    'auroc_entropy_ham_vs_pad': float(roc_auc_score(target, scores)), 'auroc_1-msp_ham_vs_pad': float(roc_auc_score(target, msp)),
    'ham_entropy_hist': np.histogram(entropy(ham_p), bins=20, range=(0, 1))[0].tolist(), 'pad_entropy_hist': np.histogram(entropy(pad_p), bins=20, range=(0, 1))[0].tolist(),
    'n_ham_test': len(ham_p), 'n_pad_all': len(pad_p),
}
print(results['shift_uncertainty'])
(RESULTS / 'exp3_cross_dataset.json').write_text(json.dumps(results, indent=1))
print('EXP3 DONE')
