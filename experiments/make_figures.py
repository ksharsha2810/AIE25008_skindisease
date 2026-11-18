"""Generates every data-driven paper figure from saved experiment results (no hand-entered numbers).
Each figure is skipped if its source result file does not exist yet."""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, RESULTS, ROOT, SHORT
from heads import classes_of

plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'ps.fonttype': 42})
COL, FULL = 3.45, 7.1
FIGURES.mkdir(parents=True, exist_ok=True)
PROPOSED = None  # set from argv: exp2 config name of the proposed disease head


def save(fig, name):
    fig.savefig(FIGURES / f'{name}.pdf')
    fig.savefig(FIGURES / f'{name}.png', dpi=200)
    plt.close(fig)
    print('wrote', name)


def fig_datasets():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.7), gridspec_kw={'width_ratios': [7, 6, 4]})
    for ax, dataset, title in zip(axes, ('HAM10000', 'PAD_UFES_20', 'ACNE04'), ('HAM10000 (dermoscopic)', 'PAD-UFES-20 (smartphone)', 'ACNE04 (facial, grades)')):
        classes = classes_of(dataset)
        counts = {part: pd.read_csv(ROOT / 'full' / 'manifests' / f'{dataset}_{part}.csv', dtype=str)['label'].value_counts() for part in ('train', 'val', 'test')}
        order = classes if dataset == 'ACNE04' else sorted(classes, key=lambda c: -sum(counts[p].get(c, 0) for p in counts))
        bottom = np.zeros(len(order))
        for part, color in zip(('train', 'val', 'test'), ('#4C72B0', '#DD8452', '#55A868')):
            values = np.array([counts[part].get(c, 0) for c in order])
            ax.bar(range(len(order)), values, bottom=bottom, color=color, label=part, width=0.75)
            bottom += values
        for i, v in enumerate(bottom):
            ax.text(i, v, str(int(v)), ha='center', va='bottom', fontsize=5.5)
        ax.set_xticks(range(len(order)), [SHORT[c] for c in order], rotation=45 if len(order) > 4 else 0)
        ax.set_title(title)
        ax.set_yscale('log' if dataset == 'HAM10000' else 'linear')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Images')
    axes[0].legend(frameon=False, loc='upper right')
    save(fig, 'fig3_dataset_distribution')


def confusion(ax, matrix, labels, title):
    matrix = np.asarray(matrix, dtype=float)
    normalized = matrix / matrix.sum(1, keepdims=True)
    ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{int(matrix[i, j])}\n{normalized[i, j]:.2f}', ha='center', va='center', fontsize=4.8 if len(labels) > 5 else 6,
                    color='white' if normalized[i, j] > 0.55 else 'black')
    ax.set_xticks(range(len(labels)), labels, rotation=0)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)


def fig_confusions():
    exp4 = json.loads((RESULTS / 'exp4_uncertainty.json').read_text())
    ham = [SHORT[c] for c in classes_of('HAM10000')]
    acne = [SHORT[c] for c in classes_of('ACNE04')]
    panels = [(exp4['deployed_disease_HAM10000']['deterministic']['confusion'], ham, '(a) Deployed head (CBAM, 512-d)')]
    if PROPOSED and PROPOSED in exp4:
        panels.append((exp4[PROPOSED]['deterministic']['confusion'], ham, '(b) GAP + 1280-d projection, CB-CE'))
    fig, axes = plt.subplots(1, len(panels), figsize=(FULL if len(panels) > 1 else COL, 2.9))
    for ax, (m, labels, title) in zip(np.atleast_1d(axes), panels):
        confusion(ax, m, labels, title)
    save(fig, 'fig5_confusion_ham10000')
    sev = [(exp4['deployed_severity_ACNE04']['deterministic']['confusion'], acne, '(a) Deployed (shared CBAM)')]
    key = next((k for k in exp4 if k.startswith('severity__')), None)
    if key:
        sev.append((exp4[key]['deterministic']['confusion'], acne, '(b) GAP + 1280-d projection'))
    fig, axes = plt.subplots(1, len(sev), figsize=(COL * len(sev) / 1.5 if len(sev) > 1 else COL * 0.8, 1.9))
    for ax, (m, labels, title) in zip(np.atleast_1d(axes), sev):
        confusion(ax, m, labels, title)
    save(fig, 'fig6_confusion_acne04')


def fig_calibration():
    exp4 = json.loads((RESULTS / 'exp4_uncertainty.json').read_text())
    keys = [('deployed_disease_HAM10000', 'Deployed')] + ([(PROPOSED, 'GAP+proj.')] if PROPOSED and PROPOSED in exp4 else [])
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3), gridspec_kw={'wspace': 0.42})
    for key, label in keys:
        rel = exp4[key]['deterministic']['reliability']
        xs = [r['conf'] for r in rel if r['count'] >= 10]
        ys = [r['acc'] for r in rel if r['count'] >= 10]
        axes[0].plot(xs, ys, 'o-', ms=3, label=f"{label} (ECE {exp4[key]['deterministic']['ece']:.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--', lw=0.7)
    axes[0].set_xlabel('Confidence'); axes[0].set_ylabel('Accuracy'); axes[0].set_title('(a) Reliability, HAM10000 test'); axes[0].legend(frameon=False, loc='upper left', fontsize=5.5)
    styles = {'1 - max softmax (det.)': ('-', 'MSP'), 'Normalized predictive entropy (MC)': ('--', 'Entropy (MC)'), 'Mutual information (MC, epistemic)': (':', 'MI (MC)')}
    for key, label in keys:
        for score, (ls, short) in styles.items():
            rc = exp4[key]['risk_coverage'][score]
            axes[1].plot(rc['curve']['coverage'], rc['curve']['risk'], ls, lw=1, label=f'{label[:3]}. {short} {rc["aurc"]:.3f}')
    axes[1].set_xlabel('Coverage'); axes[1].set_ylabel('Selective risk (error)'); axes[1].set_title('(b) Risk-coverage (AURC)'); axes[1].legend(frameon=False, fontsize=4.8, loc='upper left')
    sweep = exp4[keys[0][0]]['review_threshold_sweep']
    axes[2].plot([s['tau'] for s in sweep], [s['flag_rate'] for s in sweep], label='Fraction referred')
    axes[2].plot([s['tau'] for s in sweep], [s['acc_unflagged'] for s in sweep], label='Accuracy of retained')
    axes[2].plot([s['tau'] for s in sweep], [s['errors_captured'] for s in sweep], label='Errors referred')
    axes[2].axvline(0.4, color='k', ls='--', lw=0.7)
    axes[2].set_xlabel(r'Entropy threshold $\tau$'); axes[2].set_title(f'(c) Referral rule ({keys[0][1]} head)'); axes[2].legend(frameon=False, loc='lower left', fontsize=5.5)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'fig9_calibration_selective')


def fig_cross_dataset():
    exp3 = json.loads((RESULTS / 'exp3_cross_dataset.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.9), gridspec_kw={'width_ratios': [1.6, 1]})
    names = ['HAM->HAM (in-domain)', 'HAM->PAD test (zero-shot)', 'PAD->PAD (in-domain)', 'PAD->HAM test (zero-shot)', 'HAM+PAD joint -> PAD test', 'HAM init, PAD fine-tune -> PAD test']
    arrow = r'$\rightarrow$'
    short = ['HAM' + arrow + '\nHAM', 'HAM' + arrow + '\nPAD', 'PAD' + arrow + '\nPAD', 'PAD' + arrow + '\nHAM', 'Joint' + arrow + '\nPAD', 'HAM-init,\nPAD-tuned']
    width = 0.38
    for offset, variant in zip((-width / 2, width / 2), ('strict', 'extended')):
        s = exp3[variant]['summary']
        axes[0].bar(np.arange(len(names)) + offset, [s[n]['balanced_accuracy'][0] for n in names], width, yerr=[s[n]['balanced_accuracy'][1] for n in names],
                    label=f'{variant} ({len(exp3[variant]["classes"])} classes)', capsize=1.5)
    axes[0].set_xticks(range(len(names)), short)
    axes[0].set_ylabel('Balanced accuracy'); axes[0].set_title('(a) Label-compatible transfer (5 seeds)'); axes[0].set_ylim(0, 1); axes[0].legend(frameon=False, ncol=2, loc='upper right')
    shift = exp3['shift_uncertainty']
    bins = np.linspace(0, 1, 21)
    axes[1].stairs(np.array(shift['ham_entropy_hist']) / shift['n_ham_test'], bins, label='HAM10000 test (in-domain)', fill=True, alpha=0.5)
    axes[1].stairs(np.array(shift['pad_entropy_hist']) / shift['n_pad_all'], bins, label='PAD-UFES-20 (shifted)', fill=True, alpha=0.5)
    axes[1].axvline(0.4, color='k', ls='--', lw=0.7)
    axes[1].set_xlabel('Normalized predictive entropy'); axes[1].set_ylabel('Fraction')
    axes[1].set_title(f'(b) Entropy under shift (AUROC {shift["auroc_entropy_ham_vs_pad"]:.2f})'); axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'fig10_cross_dataset')


def fig_backbones():
    cmp = pd.read_csv(RESULTS / 'exp1_comparison_raw.csv')
    cx = json.loads((RESULTS / 'exp7_complexity.json').read_text())
    ham = cmp[cmp.Dataset == 'HAM10000'].sort_values('F1')
    pad = cmp[cmp.Dataset == 'PAD_UFES_20'].set_index('Model')
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    y = np.arange(len(ham))
    ax.barh(y + 0.19, ham.F1, 0.38, color='#4C72B0', label='HAM10000 (7 classes)')
    pad_values = [pad.F1.get(model, np.nan) for model in ham.Model]
    ax.barh(y - 0.19, pad_values, 0.38, color='#DD8452', label='PAD-UFES-20 (6 classes)')
    labels = []
    for model, name in zip(ham.Model, ham.Name):
        c = cx[f'{model}|7']
        labels.append(f"{name}\n{c['params_M']:.2f} M | {c['GMACs']:.2f} GMACs | {c['latency_median_ms']:.1f} ms")
    ax.set_yticks(y, labels, fontsize=6.6)
    for yi, v in zip(y, ham.F1):
        ax.text(v + 0.003, yi + 0.19, f'{100 * v:.1f}', va='center', fontsize=6.4)
    for yi, v in zip(y, pad_values):
        if not np.isnan(v):
            ax.text(v + 0.003, yi - 0.19, f'{100 * v:.1f}', va='center', fontsize=6.4)
    ax.set_xlim(0.0, 0.66)
    ax.set_xlabel('Test macro-F1 (frozen-feature linear probe)')
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=6.6)
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'fig4_backbone_tradeoff')


def fig_robustness():
    df = pd.read_csv(RESULTS / 'exp5_robustness.csv')
    heads = list(df['head'].unique())
    kinds = [k for k in df['corruption'].unique() if k != 'clean']
    fig, axes = plt.subplots(2, len(kinds), figsize=(FULL, 2.6), sharey='row')
    for j, kind in enumerate(kinds):
        for head in heads:
            clean = df[(df['head'] == head) & (df.corruption == 'clean')]
            sub = pd.concat([clean, df[(df['head'] == head) & (df.corruption == kind)]]).sort_values('severity')
            axes[0, j].plot(sub.severity, sub.macro_f1, 'o-', ms=2, lw=0.9, label=head)
            axes[1, j].plot(sub.severity, sub.mean_entropy, 'o-', ms=2, lw=0.9)
        axes[0, j].set_title(kind.replace('_', ' '), fontsize=6.5)
        axes[1, j].set_xlabel('Severity')
        axes[0, j].set_xticks([0, 1, 2, 3]); axes[1, j].set_xticks([0, 1, 2, 3])
    axes[0, 0].set_ylabel('Macro-F1'); axes[1, 0].set_ylabel('Mean entropy')
    axes[0, 0].legend(frameon=False, fontsize=5, loc='lower left')
    save(fig, 'fig11_robustness')


def fig_gradcam():
    examples = np.load(RESULTS / 'exp6_examples.npy', allow_pickle=True).item()
    names = [n for n in examples if n.startswith('Proposed')] or list(examples)
    items = examples[names[0]][:6]
    labels = classes_of('HAM10000')
    fig, axes = plt.subplots(2, len(items), figsize=(FULL, 2.6))
    for j, item in enumerate(items):
        axes[0, j].imshow(item['rgb']); axes[0, j].contour(item['mask'], levels=[0.5], colors='lime', linewidths=0.6)
        axes[0, j].set_title(f"GT {SHORT[labels[item['label']]]} / Pred {SHORT[labels[item['pred']]]}", fontsize=6)
        axes[1, j].imshow(item['rgb']); axes[1, j].imshow(item['cam'].astype(np.float32), cmap='jet', alpha=0.45); axes[1, j].contour(item['mask'], levels=[0.5], colors='white', linewidths=0.6)
        axes[1, j].set_title(f"p={item['conf']:.2f}, H={item['entropy']:.2f}", fontsize=6)
        for ax in axes[:, j]:
            ax.axis('off')
    save(fig, 'fig7_gradcam_examples')


if __name__ == '__main__':
    PROPOSED = sys.argv[1] if len(sys.argv) > 1 else None
    for function in (fig_datasets, fig_confusions, fig_calibration, fig_cross_dataset, fig_backbones, fig_robustness, fig_gradcam):
        try:
            function()
        except FileNotFoundError as error:
            print('skip', function.__name__, '-', error.filename)


