"""Export the deployed dual-head model (frozen PP-LCNet-1.0 + CBAM + lesion/acne heads) for the
browser demo in docs/.

The ONNX graph returns the CBAM-refined feature map (512x7x7) and its global-average-pooled
vector (512). The two linear heads, Monte Carlo dropout (T=30, p=0.3) and the class activation
map are computed in JavaScript from the exported head weights, so the browser reproduces the
deployed pipeline exactly, including MC-dropout sampling.

Usage:
    python demo/export_onnx.py --checkpoint <path>/dual_final.pt --out docs/model
"""
import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn

HAM_CLASSES = ['actinic_keratosis_or_intraepithelial_carcinoma', 'basal_cell_carcinoma', 'benign_keratosis', 'dermatofibroma', 'melanoma', 'nevus', 'vascular_lesion']
ACNE_GRADES = ['grade_1', 'grade_2', 'grade_3', 'grade_4']


class CBAM(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(), nn.Conv2d(hidden, channels, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        x = x * torch.sigmoid(self.channel_mlp(x.mean((2, 3), keepdim=True)) + self.channel_mlp(x.amax((2, 3), keepdim=True)))
        return x * torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))


class FeatureExtractor(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.backbone = timm.create_model('lcnet_100.ra2_in1k', pretrained=False, num_classes=7)
        self.backbone.load_state_dict({k[len('backbone.'):]: v for k, v in state.items() if k.startswith('backbone.')})
        self.cbam = CBAM(512)
        self.cbam.load_state_dict({k[len('cbam.'):]: v for k, v in state.items() if k.startswith('cbam.')})

    def forward(self, images):
        refined = self.cbam(self.backbone.forward_features(images))
        return refined, refined.mean((2, 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', default='docs/model')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['state']
    model = FeatureExtractor(state).eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy, out / 'lcnet_cbam.onnx', input_names=['image'], output_names=['refined', 'pooled'], opset_version=17, dynamo=False)
    cfg = timm.get_pretrained_cfg('lcnet_100.ra2_in1k')
    heads = {
        'lesion': {'classes': HAM_CLASSES, 'weight': state['disease_head.weight'].tolist(), 'bias': state['disease_head.bias'].tolist()},
        'acne': {'classes': ACNE_GRADES, 'weight': state['severity_head.weight'].tolist(), 'bias': state['severity_head.bias'].tolist()},
        'dropout': 0.3, 'mc_passes': 30, 'referral_entropy_threshold': 0.4,
        'preprocess': {'size': 224, 'mean': list(cfg.mean), 'std': list(cfg.std), 'resize': 'shorter side to 224, centre crop'},
    }
    (out / 'heads.json').write_text(json.dumps(heads))

    import onnxruntime as ort
    session = ort.InferenceSession(str(out / 'lcnet_cbam.onnx'))
    sample = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        refined, pooled = model(sample)
    o_refined, o_pooled = session.run(None, {'image': sample.numpy()})
    print('max |torch - onnx| refined', float(np.abs(refined.numpy() - o_refined).max()), 'pooled', float(np.abs(pooled.numpy() - o_pooled).max()))
    print('written', out / 'lcnet_cbam.onnx', (out / 'lcnet_cbam.onnx').stat().st_size // 1024, 'KB')


if __name__ == '__main__':
    main()
