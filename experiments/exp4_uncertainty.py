"""Experiment 4: uncertainty, calibration and selective prediction.

Evaluates the *deployed* dual-head checkpoint (dual_final.pt: frozen LCNet-1.0 + CBAM +
disease/severity heads) on the full HAM10000 and ACNE04 test sets, plus any extra configs
passed by name from exp2 (e.g. the proposed head). Reports deterministic vs 30-pass MC
dropout, temperature scaling, ECE/MCE/Brier/NLL, risk-coverage/AURC, misclassification
AUROC of each uncertainty score, and audits the project's 0.4 normalized-entropy review rule.
"""
import json
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from common import RESULTS, ROOT
from heads import CBAM, classes_of, ece_score, full_metrics, load_cached

torch.set_num_threads(4)
T_PASSES = 30


class Deployed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cbam = CBAM(512)
        self.dropout = torch.nn.Dropout(0.3)
        self.disease_head = torch.nn.Linear(512, 7)
        self.severity_head = torch.nn.Linear(512, 4)

    def forward(self, x, task):
        return getattr(self, f'{task}_head')(self.dropout(self.cbam(x).mean((2, 3))))


def deployed_outputs(task, dataset):
    model = Deployed()
    state = torch.load(ROOT / 'full' / 'dual_final.pt', map_location='cpu', weights_only=True)['state']
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith('backbone.')})
    model.eval()
    out = {}
    for part in ('val', 'test'):
        x, y = load_cached(dataset, part)
        with torch.no_grad():
            logits = torch.cat([model(x[i:i + 512], task) for i in range(0, len(x), 512)])
            torch.manual_seed(42)
            model.dropout.train()
            mc = torch.stack([torch.cat([model(x[i:i + 512], task).softmax(1) for i in range(0, len(x), 512)]) for _ in range(T_PASSES)])
            model.dropout.eval()
        out[part] = {'logits': logits.numpy(), 'mc': mc.numpy(), 'y': y.numpy()}
    return out


def fit_temperature(logits, y):
    t = torch.zeros(1, requires_grad=True)
    lg, yy = torch.tensor(logits), torch.tensor(y)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / t.exp(), yy)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.exp())


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def norm_entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(1) / np.log(p.shape[1])


def mce(p, y, bins=15):
    conf, correct = p.max(1), (p.argmax(1) == y)
    edges = np.linspace(0, 1, bins + 1)
    gaps = [abs(correct[(conf > a) & (conf <= b)].mean() - conf[(conf > a) & (conf <= b)].mean()) for a, b in zip(edges[:-1], edges[1:]) if ((conf > a) & (conf <= b)).any()]
    return float(max(gaps))


def reliability(p, y, bins=10):
    conf, correct = p.max(1), (p.argmax(1) == y)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        rows.append({'lo': float(a), 'hi': float(b), 'count': int(m.sum()), 'acc': float(correct[m].mean()) if m.any() else None, 'conf': float(conf[m].mean()) if m.any() else None})
    return rows


def risk_coverage(score, p, y):
    """score: higher = more uncertain. Returns curve and AURC (selective risk = 1-accuracy)."""
    order = np.argsort(score)
    correct = (p.argmax(1) == y)[order]
    n = len(y)
    coverage = np.arange(1, n + 1) / n
    risk = 1 - np.cumsum(correct) / np.arange(1, n + 1)
    at = {}
    for c in (0.5, 0.7, 0.8, 0.9, 1.0):
        k = int(round(c * n))
        idx = order[:k]
        at[str(c)] = {'accuracy': float((p.argmax(1)[idx] == y[idx]).mean()), 'macro_f1': float(f1_score(y[idx], p.argmax(1)[idx], average='macro', labels=list(range(p.shape[1])), zero_division=0))}
    # optimal (oracle) AURC for reference
    oracle = np.sort(~(p.argmax(1) == y))
    oracle_risk = np.cumsum(oracle) / np.arange(1, n + 1)
    step = max(1, n // 200)
    return {'aurc': float(risk.mean()), 'oracle_aurc': float(oracle_risk.mean()), 'eaurc': float(risk.mean() - oracle_risk.mean()), 'at_coverage': at,
            'curve': {'coverage': coverage[::step].tolist(), 'risk': risk[::step].tolist()}}


def analyse(name, out, num_classes, ordinal=False):
    val, test = out['val'], out['test']
    y = test['y']
    det = softmax(test['logits'])
    mc = test['mc']
    mc_mean = mc.mean(0)
    temperature = fit_temperature(val['logits'], val['y'])
    ts = softmax(test['logits'] / temperature)
    expected_entropy = np.mean([norm_entropy(m) for m in mc], 0)
    total_entropy = norm_entropy(mc_mean)
    mutual_info = total_entropy - expected_entropy
    wrong = (mc_mean.argmax(1) != y).astype(int)
    scores = {'1 - max softmax (det.)': 1 - det.max(1), 'Normalized predictive entropy (MC)': total_entropy, 'Expected entropy (MC, aleatoric)': expected_entropy,
              'Mutual information (MC, epistemic)': mutual_info, '1 - max prob (MC)': 1 - mc_mean.max(1), '1 - max prob (TS)': 1 - ts.max(1)}
    result = {'name': name, 'temperature': temperature, 'n_test': int(len(y))}
    for label, p in (('deterministic', det), ('mc_dropout', mc_mean), ('temperature_scaled', ts)):
        metrics = full_metrics(p, y, num_classes, ordinal)
        metrics['mce'] = mce(p, y)
        metrics['reliability'] = reliability(p, y)
        result[label] = metrics
    result['misclassification_auroc'] = {k: float(roc_auc_score((({'1 - max softmax (det.)': det, '1 - max prob (TS)': ts}.get(k, mc_mean)).argmax(1) != y).astype(int), s)) for k, s in scores.items()}
    result['risk_coverage'] = {k: risk_coverage(s, {'1 - max softmax (det.)': det, '1 - max prob (TS)': ts}.get(k, mc_mean), y) for k, s in scores.items()}
    flagged = total_entropy > 0.4
    pred = mc_mean.argmax(1)
    result['review_rule_0.4'] = {
        'fraction_flagged': float(flagged.mean()),
        'accuracy_unflagged': float((pred[~flagged] == y[~flagged]).mean()) if (~flagged).any() else None,
        'accuracy_flagged': float((pred[flagged] == y[flagged]).mean()) if flagged.any() else None,
        'errors_captured_fraction': float(flagged[wrong == 1].mean()) if wrong.any() else None,
        'per_class_flag_rate': [float(flagged[y == k].mean()) for k in range(num_classes)],
    }
    thresholds = np.linspace(0.05, 0.95, 19)
    result['review_threshold_sweep'] = [{'tau': float(t), 'flag_rate': float((total_entropy > t).mean()), 'acc_unflagged': float((pred[total_entropy <= t] == y[total_entropy <= t]).mean()) if (total_entropy <= t).any() else None,
                                         'errors_captured': float((total_entropy > t)[wrong == 1].mean())} for t in thresholds]
    result['mean_entropy_correct_vs_wrong'] = [float(total_entropy[wrong == 0].mean()), float(total_entropy[wrong == 1].mean())]
    result['mean_entropy_per_class'] = [float(total_entropy[y == k].mean()) for k in range(num_classes)]
    result['mc_vs_det_prediction_agreement'] = float((det.argmax(1) == pred).mean())
    np.savez_compressed(RESULTS / f'exp4_{name}_arrays.npz', det=det, mc_mean=mc_mean, ts=ts, y=y, total_entropy=total_entropy, mutual_info=mutual_info)
    return result


def from_exp2(name, seed=0):
    data = np.load(RESULTS / 'exp2' / f'{name}__seed{seed}.npz')
    # exp2 stores probabilities; recover log-probabilities as logits for TS (softmax invariant to shift)
    return {'val': {'logits': np.log(np.clip(data['val'], 1e-12, 1)), 'y': data['yval'], 'mc': None},
            'test': {'logits': np.log(np.clip(data['test'], 1e-12, 1)), 'y': data['ytest'], 'mc': (lambda a: a / a.sum(-1, keepdims=True))(data['test_mc'].astype(np.float64))}}


if __name__ == '__main__':
    report = {}
    disease = deployed_outputs('disease', 'HAM10000')
    report['deployed_disease_HAM10000'] = analyse('deployed_disease', disease, 7)
    severity = deployed_outputs('severity', 'ACNE04')
    report['deployed_severity_ACNE04'] = analyse('deployed_severity', severity, 4, ordinal=True)
    for name in sys.argv[1:]:
        task = 'severity' if name.startswith('severity') else 'disease'
        report[name] = analyse(name.replace('/', '_'), from_exp2(name), 4 if task == 'severity' else 7, ordinal=task == 'severity')
    for key, value in report.items():
        print(key, 'T=', round(value['temperature'], 3), {m: {k: round(value[m][k], 4) for k in ('accuracy', 'macro_f1', 'ece', 'brier', 'nll')} for m in ('deterministic', 'mc_dropout', 'temperature_scaled')})
        print('  misclf AUROC', {k: round(v, 3) for k, v in value['misclassification_auroc'].items()})
        print('  AURC', {k: round(v['aurc'], 4) for k, v in value['risk_coverage'].items()})
        print('  review rule', value['review_rule_0.4'])
    (RESULTS / 'exp4_uncertainty.json').write_text(json.dumps(report, indent=1))
    print('EXP4 DONE')
