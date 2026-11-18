"""Shared setup for the paper experiments.

Reuses the project's runtime (``skin_pipeline_runtime.py``) unchanged, but runs in an
isolated root (``paper_experiments/root``) so the original artifacts are never modified.
"""
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
PROJECT = Path(os.environ.get('SKIN_PROJECT_ROOT', r'C:\clg'))
EXP = PROJECT / 'paper_experiments'
ROOT = EXP / 'root'
RESULTS = EXP / 'results'
FIGURES = EXP / 'figures'
RUNTIME = Path(__file__).resolve().parents[1] / 'src' / 'skin_pipeline_runtime.py'
CACHE = PROJECT / '.cache'
os.environ.setdefault('TORCH_HOME', str(CACHE / 'torch'))
os.environ.setdefault('HF_HOME', str(CACHE / 'huggingface'))
os.environ.setdefault('HF_HUB_CACHE', str(CACHE / 'huggingface' / 'hub'))
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('MPLCONFIGDIR', str(CACHE / 'matplotlib'))
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

EXTRA_MODELS = ('efficientnet_b0.ra_in1k', 'mobilenetv2_100.ra_in1k', 'ghostnet_100.in1k', 'mobilevit_xxs.cvnets_in1k')
EXTRA_DISPLAY = ('EfficientNet-B0', 'MobileNetV2-1.0', 'GhostNet-1.0', 'MobileViT-XXS')
DISEASE_DATASETS = ('HAM10000', 'PAD_UFES_20')
SHORT = {
    'actinic_keratosis_or_intraepithelial_carcinoma': 'AKIEC', 'basal_cell_carcinoma': 'BCC', 'benign_keratosis': 'BKL',
    'dermatofibroma': 'DF', 'melanoma': 'MEL', 'nevus': 'NV', 'vascular_lesion': 'VASC', 'actinic_keratosis': 'ACK',
    'seborrheic_keratosis': 'SEK', 'squamous_cell_carcinoma': 'SCC', 'grade_1': 'G1', 'grade_2': 'G2', 'grade_3': 'G3', 'grade_4': 'G4',
}


def load_runtime(extended=True):
    spec = importlib.util.spec_from_file_location('skin_pipeline_runtime', RUNTIME)
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)
    if extended:
        runtime.MODELS = tuple(runtime.MODELS) + EXTRA_MODELS
        runtime.DISPLAY = tuple(runtime.DISPLAY) + EXTRA_DISPLAY
    return runtime


def pipeline(extended=True):
    runtime = load_runtime(extended)
    return runtime, runtime.Pipeline(ROOT)


def save(name, value):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
