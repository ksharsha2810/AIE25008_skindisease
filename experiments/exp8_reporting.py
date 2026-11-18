"""Experiment 8: reporting reliability of the LLM formatter.

Stage A (offline, runs without an API key): builds evidence payloads for a stratified sample
of HAM10000 test cases and validates the automatic report checker with fault injection
(every injected fault type must be detected; every untouched reference report must pass).

Stage B (requires GEMINI_API_KEY, run by the author): sends each payload R times to Gemini
under two prompting regimes and scores
  - numeric fidelity        every number in the text equals a payload value (4 d.p.)
  - label fidelity          predicted class / grade / applicability / review flag preserved
  - unsupported-claim rate  sentences containing forbidden clinical content (lexicon) or
                            numbers/classes not present in the payload
  - omission rate           required facts (class, confidence, entropy, review flag) missing
  - consistency             exact agreement of the R generations
Regimes: 'template' = the project's closed six-sentence template (exact-match guard);
         'free'     = JSON with free-form sentences constrained only by the system policy.
Usage: python exp8_reporting.py offline | python exp8_reporting.py gemini [model] [repeats]
"""
import copy
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

from common import RESULTS, SHORT
from heads import classes_of, load_cached
from models import DeployedHead

FORBIDDEN = re.compile(r'\b(diagnos\w*|treat\w*|prescri\w*|surg\w*|excis\w*|biops\w*|benign|malignan\w*|cancer\w*|reassur\w*|no need|safe to|definitely|certainly|harmless)\b', re.I)
DISCLAIMERS = ('This is a research prediction, not a clinician diagnosis.', 'clinical validation is still required.')
NUMBER = re.compile(r'\d+\.\d+')
OUT = RESULTS / 'exp8'


def sanitize(sentence):
    """Remove the mandated disclaimer and dataset class identifiers before lexicon screening."""
    for phrase in DISCLAIMERS:
        sentence = sentence.replace(phrase, ' ')
    for name in classes_of('HAM10000'):
        sentence = sentence.replace(name, ' ').replace(name.replace('_', ' '), ' ')
    return sentence


def norm_entropy(p):
    return float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(len(p)))


def build_payloads(n_per_class=8, seed=0):
    classes = classes_of('HAM10000')
    x, y = load_cached('HAM10000', 'test')
    head = DeployedHead().eval()
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([rng.choice(np.where(y.numpy() == k)[0], min(n_per_class, int((y == k).sum())), replace=False) for k in range(len(classes))])
    payloads = []
    for index in chosen:
        with torch.no_grad():
            torch.manual_seed(42)
            head.dropout.train()
            mc = torch.stack([head(x[index:index + 1]).softmax(1)[0] for _ in range(30)]).mean(0).numpy()
        entropy = norm_entropy(mc)
        payloads.append({'schema_version': '1.0', 'case_id': f'ham_test_{int(index)}', 'mode': 'RESEARCH_NOT_CLINICAL',
                         'disease': {'class': classes[int(mc.argmax())], 'confidence': round(float(mc.max()), 4), 'normalized_entropy': round(entropy, 4),
                                     'top2': [{'class': classes[i], 'probability': round(float(mc[i]), 4)} for i in np.argsort(-mc)[:2]]},
                         'specialist_review_required': bool(entropy > 0.4),
                         'review_reasons': ['Disease normalized predictive entropy exceeds 0.4.'] if entropy > 0.4 else [],
                         '_ground_truth_for_analysis_only': classes[int(y[index])]})
    return payloads


def public(payload):
    return {k: v for k, v in payload.items() if not k.startswith('_')}


def allowed_numbers(payload):
    values = {payload['disease']['confidence'], payload['disease']['normalized_entropy'], 0.4}
    values |= {item['probability'] for item in payload['disease']['top2']}
    return {f'{v:.4f}' for v in values} | {f'{v:.1f}' for v in values} | {f'{v:.2f}' for v in values}


def score_report(report, payload):
    """Returns a dict of booleans/counts. Report: {'sentences': [...], 'specialist_review_required': bool}."""
    text = ' '.join(report.get('sentences', []))
    allowed = allowed_numbers(payload)
    numbers = NUMBER.findall(text)
    sentences = report.get('sentences', [])
    classes = set(classes_of('HAM10000'))
    mentioned_classes = {c for c in classes if c in text or c.replace('_', ' ') in text.lower()}
    payload_classes = {payload['disease']['class']} | {t['class'] for t in payload['disease']['top2']}
    unsupported = [s for s in sentences if FORBIDDEN.search(sanitize(s)) or any(n not in allowed for n in NUMBER.findall(s))]
    return {
        'numeric_fidelity': all(n in allowed for n in numbers),
        'class_fidelity': payload['disease']['class'] in text and mentioned_classes <= payload_classes,
        'flag_fidelity': report.get('specialist_review_required') == payload['specialist_review_required'] and (('review' in text.lower()) if payload['specialist_review_required'] else True),
        'omission': not (payload['disease']['class'] in text and f"{payload['disease']['confidence']:.4f}" in text and f"{payload['disease']['normalized_entropy']:.4f}" in text),
        'unsupported_sentences': len(unsupported), 'n_sentences': len(sentences),
    }


def template_sentences(payload):
    d = payload['disease']
    return [f"Model disease class: {d['class']}. This is a research prediction, not a clinician diagnosis.",
            f"MC-dropout confidence: {d['confidence']:.4f}; normalized predictive entropy: {d['normalized_entropy']:.4f}.",
            f"Second most probable class: {d['top2'][1]['class']} ({d['top2'][1]['probability']:.4f}).",
            'Specialist review required.' if payload['specialist_review_required'] else 'Uncertainty threshold does not trigger specialist review; clinical validation is still required.']


FAULTS = {
    'altered_number': lambda r, p: {**r, 'sentences': [s.replace(f"{p['disease']['confidence']:.4f}", f"{min(0.9999, p['disease']['confidence'] + 0.05):.4f}") for s in r['sentences']]},
    'swapped_class': lambda r, p: {**r, 'sentences': [s.replace(p['disease']['class'], p['disease']['top2'][1]['class'], 1) if i == 0 else s for i, s in enumerate(r['sentences'])]},
    'flipped_flag': lambda r, p: {**r, 'specialist_review_required': not r['specialist_review_required']},
    'dropped_uncertainty': lambda r, p: {**r, 'sentences': [s for s in r['sentences'] if 'entropy' not in s]},
    'added_treatment_claim': lambda r, p: {**r, 'sentences': r['sentences'] + ['Treatment with topical therapy is recommended.']},
    'added_reassurance': lambda r, p: {**r, 'sentences': r['sentences'] + ['The lesion is benign and there is no need for follow-up.']},
    'invented_number': lambda r, p: {**r, 'sentences': r['sentences'] + ['Lesion diameter is approximately 6.5 mm.']},
}


def detected(scores):
    return (not scores['numeric_fidelity']) or (not scores['class_fidelity']) or (not scores['flag_fidelity']) or scores['omission'] or scores['unsupported_sentences'] > 0


def offline():
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = build_payloads()
    (OUT / 'payloads.json').write_text(json.dumps(payloads, indent=1))
    clean_pass, table = 0, {name: 0 for name in FAULTS}
    for payload in payloads:
        reference = {'sentences': template_sentences(payload), 'specialist_review_required': payload['specialist_review_required']}
        clean_pass += not detected(score_report(reference, payload))
        for name, fault in FAULTS.items():
            table[name] += detected(score_report(fault(copy.deepcopy(reference), payload), payload))
    n = len(payloads)
    flags = np.mean([p['specialist_review_required'] for p in payloads])
    result = {'n_payloads': n, 'flag_rate': float(flags), 'reference_reports_passing': clean_pass / n, 'fault_detection_rate': {k: v / n for k, v in table.items()}}
    print(json.dumps(result, indent=1))
    (RESULTS / 'exp8_offline_checker.json').write_text(json.dumps(result, indent=1))


POLICY = ('You are a fact-only formatter, never a diagnostician. You receive structured model outputs only, never an image. '
          'Do not add diagnoses, findings, lesion descriptions, treatment, reassurance or severity interpretation. Reproduce numbers exactly to four decimals. '
          'If specialist_review_required is true, state that specialist review is required. Return JSON with keys sentences (list of strings) and specialist_review_required (boolean).')


def gemini(model_name='gemini-2.5-flash', repeats=3):
    import google.generativeai as genai
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        raise SystemExit('Set GEMINI_API_KEY to run Stage B. No results are fabricated.')
    genai.configure(api_key=key, transport='rest')
    payloads = json.loads((OUT / 'payloads.json').read_text())
    rows = []
    for regime in ('template', 'free'):
        for payload in payloads:
            facts = public(payload)
            system = POLICY + (' Use exactly these sentences: ' + json.dumps(template_sentences(payload)) if regime == 'template' else ' Write 3-5 short sentences.')
            model = genai.GenerativeModel(model_name, system_instruction=system)
            outputs = []
            for r in range(repeats):
                try:
                    response = model.generate_content(json.dumps(facts), generation_config={'temperature': 0, 'response_mime_type': 'application/json'})
                    report = json.loads(response.text)
                    outputs.append(report)
                    rows.append({'regime': regime, 'case': payload['case_id'], 'repeat': r, 'parse_ok': True, **score_report(report, payload), 'exact_template': report.get('sentences') == template_sentences(payload)})
                except Exception as error:
                    rows.append({'regime': regime, 'case': payload['case_id'], 'repeat': r, 'parse_ok': False, 'error': type(error).__name__})
            if outputs:
                rows[-1]['consistent_across_repeats'] = all(o == outputs[0] for o in outputs)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f'exp8_gemini_{model_name}.csv', index=False)
    summary = df.groupby('regime').agg(parse_ok=('parse_ok', 'mean'), numeric_fidelity=('numeric_fidelity', 'mean'), class_fidelity=('class_fidelity', 'mean'), flag_fidelity=('flag_fidelity', 'mean'),
                                       omission=('omission', 'mean'), unsupported=('unsupported_sentences', 'sum'), sentences=('n_sentences', 'sum'), consistent=('consistent_across_repeats', 'mean'))
    summary['unsupported_claim_rate'] = summary['unsupported'] / summary['sentences']
    print(summary.to_string())
    summary.to_csv(RESULTS / f'exp8_gemini_{model_name}_summary.csv')


if __name__ == '__main__':
    if sys.argv[1] == 'offline':
        offline()
    else:
        gemini(*(sys.argv[2:3] or ['gemini-2.5-flash']), *([int(sys.argv[3])] if len(sys.argv) > 3 else []))
