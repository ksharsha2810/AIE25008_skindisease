"""Image-level models (frozen LCNet-1.0 backbone + different heads) for robustness,
explainability and complexity experiments."""
import json

import numpy as np
import timm
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from common import RESULTS, ROOT
from heads import CBAM, Head, load_cached, pretrained_conv_head, train

BACKBONE = 'lcnet_100.ra2_in1k'
LCNET_CKPT = ROOT / 'full' / 'HAM10000_lcnet_100.ra2_in1k.pt'


def backbone():
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=7)
    model.load_state_dict(torch.load(LCNET_CKPT, map_location='cpu', weights_only=True)['state'])
    return model.eval()


def geometry():
    cfg = timm.get_pretrained_cfg(BACKBONE)
    interpolation = {'bicubic': transforms.InterpolationMode.BICUBIC, 'bilinear': transforms.InterpolationMode.BILINEAR}[cfg.interpolation]
    return transforms.Compose([transforms.Resize(224, interpolation=interpolation), transforms.CenterCrop(224)]), cfg


def normalizer():
    _, cfg = geometry()
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)])


class DeployedHead(nn.Module):
    """The project's deployed dual-head (dual_final.pt), disease branch."""
    def __init__(self):
        super().__init__()
        self.cbam = CBAM(512)
        self.dropout = nn.Dropout(0.3)
        self.disease_head = nn.Linear(512, 7)
        self.severity_head = nn.Linear(512, 4)
        state = torch.load(ROOT / 'full' / 'dual_final.pt', map_location='cpu', weights_only=True)['state']
        self.load_state_dict({k: v for k, v in state.items() if not k.startswith('backbone.')})

    def forward(self, x):
        return self.disease_head(self.dropout(self.cbam(x).mean((2, 3))))


class LinearProbeHead(nn.Module):
    """Benchmark LCNet linear probe: GAP -> pretrained conv_head -> hardswish -> classifier."""
    def __init__(self):
        super().__init__()
        state = torch.load(LCNET_CKPT, map_location='cpu', weights_only=True)['state']
        self.conv_head = nn.Conv2d(512, 1280, 1)
        self.conv_head.load_state_dict({'weight': state['conv_head.weight'], 'bias': state['conv_head.bias']})
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1280, 7)
        self.fc.load_state_dict({'weight': state['classifier.weight'], 'bias': state['classifier.bias']})

    def forward(self, x):
        return self.fc(self.dropout(nn.functional.hardswish(self.conv_head(x.mean((2, 3), keepdim=True))).flatten(1)))


def trained_head(attention, convhead, loss, seed=0):
    """Re-trains an exp2 configuration deterministically (same seed path as exp2)."""
    from exp2_heads import build
    path = RESULTS / 'heads' / f'{attention}{"+convhead" if convhead else ""}__{loss}__seed{seed}.pt'
    torch.manual_seed(seed)
    model = build(attention, convhead, 7)
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.eval()
    data = [load_cached('HAM10000', part) for part in ('train', 'val')]
    model, _ = train(model, *data[0], *data[1], 7, loss=loss, epochs=30, patience=8, seed=seed, select='f1')
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model.eval()


class Full(nn.Module):
    def __init__(self, trunk, head):
        super().__init__()
        self.trunk, self.head = trunk, head

    def forward(self, images):
        return self.head(self.trunk.forward_features(images))


def heads_under_test(proposed):
    heads = {'Deployed (CBAM, GAP-512)': DeployedHead(), 'Linear probe (no attention)': LinearProbeHead()}
    if proposed:
        att, convhead, loss = proposed
        heads[f'Proposed ({att}{"+convhead" if convhead else ""}, {loss})'] = trained_head(att, convhead, loss)
    for head in heads.values():
        head.eval()
    return heads


def parse_proposed(argv):
    if len(argv) < 2 or not argv[1] or argv[1] == 'none':
        return None
    att, convhead, loss = argv[1].split(',')
    return att, convhead == 'convhead', loss
