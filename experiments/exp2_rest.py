"""Remaining disease-head ablation configurations: 2 seeds, 15 epochs, patience 4."""
import sys

import torch

from exp2_heads import run
from heads import load_cached

torch.set_num_threads(2)
CONFIGS = [('cbam_spatial', False, 'ce'), ('se', False, 'ce'), ('eca', False, 'ce'), ('none', True, 'ce'), ('cbam', True, 'ce'),
           ('none', True, 'wce'), ('none', True, 'la'), ('none', True, 'focal'), ('none', True, 'ls'),
           ('cbam', False, 'wce'), ('cbam', False, 'la'), ('cbam', True, 'wce'), ('cbam', True, 'la'), ('cbam', False, 'focal'), ('cbam', False, 'ls')]
worker, workers = int(sys.argv[1]), int(sys.argv[2])
data = [load_cached('HAM10000', part) for part in ('train', 'val', 'test')]
for i, (att, convhead, loss) in enumerate(CONFIGS):
    if i % workers == worker:
        name = f'disease__{att}{"+convhead" if convhead else ""}__{loss}__sel-f1__ep15'
        run('disease', 'HAM10000', att, convhead, loss, 'f1', 15, [0, 1], data, tag=name, patience=4)
print('WORKER DONE', worker)
