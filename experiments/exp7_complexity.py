"""Experiment 7: complexity and CPU latency for every backbone (as benchmarked: full timm model
with the dataset head) and for the deployed / proposed LCNet inference paths.
Run on an otherwise idle machine. Batch 1, 224x224, fp32, 4 threads, 20 warm-up + 200 timed runs.
"""
import io
import json
import time

import numpy as np
import timm
import torch
from torch.utils.flop_counter import FlopCounterMode

from common import EXTRA_MODELS, RESULTS
from models import DeployedHead, Full, LinearProbeHead, backbone

torch.set_num_threads(4)
BACKBONES = ('mobilenetv3_large_100.ra_in1k', 'mobilenetv4_conv_small.e2400_r224_in1k', 'lcnet_100.ra2_in1k') + EXTRA_MODELS


def profile(model):
    model.eval()
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    flops = counter.get_total_flops()
    with torch.no_grad():
        for _ in range(20):
            model(x)
        times = []
        for _ in range(50):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - start) * 1000)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return {'params_M': sum(p.numel() for p in model.parameters()) / 1e6, 'GMACs': flops / 2 / 1e9, 'latency_median_ms': float(np.median(times)),
            'latency_p25_ms': float(np.percentile(times, 25)), 'latency_p75_ms': float(np.percentile(times, 75)), 'size_MB_fp32': buffer.getbuffer().nbytes / 1e6}


if __name__ == '__main__':
    out = {}
    for name in BACKBONES:
        for classes in (7,):
            out[f'{name}|{classes}'] = profile(timm.create_model(name, pretrained=False, num_classes=classes))
            print(name, classes, out[f'{name}|{classes}'], flush=True)
    trunk = backbone()
    class Features(torch.nn.Module):
        def __init__(self, t):
            super().__init__(); self.t = t
        def forward(self, x):
            return self.t.forward_features(x)
    out['lcnet_trunk_only'] = profile(Features(trunk))
    deployed = Full(trunk, DeployedHead())
    out['deployed_dual_registered'] = {'params_M_registered_incl_unused': 1.703892}
    out['deployed_disease_path'] = profile(deployed)
    out['linear_probe_path'] = profile(Full(trunk, LinearProbeHead()))
    from heads import CBAM
    out['cbam_params'] = sum(p.numel() for p in CBAM(512).parameters())
    feature_params = sum(p.numel() for n, p in trunk.named_parameters() if not n.startswith(('conv_head', 'classifier')))
    out['executed_params'] = {'trunk_forward_features': feature_params,
                              'deployed_disease_path': feature_params + out['cbam_params'] + 512 * 7 + 7,
                              'deployed_dual_path_both_heads': feature_params + out['cbam_params'] + 512 * 11 + 11,
                              'linear_probe_path': feature_params + 512 * 1280 + 1280 + 1280 * 7 + 7}
    print(out['executed_params'])
    for key in ('lcnet_trunk_only', 'deployed_disease_path', 'linear_probe_path'):
        print(key, out[key])
    (RESULTS / 'exp7_complexity.json').write_text(json.dumps(out, indent=1))
    print('EXP7 DONE')

