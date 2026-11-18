"""PAD-UFES-20 linear probe for one backbone (same protocol as exp1, seed 42)."""
import json
import sys

import numpy as np
import torch

from common import RESULTS, pipeline

torch.set_num_threads(2)
name = sys.argv[1]
runtime, pipe = pipeline(extended=True)
dataset = 'PAD_UFES_20'
pipe.seed()
path = pipe.run / f'{dataset}_{name}.pt'
classes = sorted(pipe.splits(dataset)[0]['label'].unique())
model = pipe.factory(name, len(classes), pretrained=not path.exists())
loaders, classes = pipe.loaders(dataset, augment=False)
if path.exists():
    model.load_state_dict(pipe.load_checkpoint(path)['state'])
else:
    pipe.fit(model, loaders, classes, path)
validation = pipe.evaluate(model, loaders[1], classes)
test = pipe.evaluate(model, loaders[2], classes)
row = {'Dataset': dataset, 'Model': name, 'Name': runtime.DISPLAY[runtime.MODELS.index(name)], **{k: v for k, v in test.items() if k != 'Confusion'}, 'ValAccuracy': validation['Accuracy']}
model.eval()
for split, loader in (('val', loaders[1]), ('test', loaders[2])):
    logits, labels = [], []
    with torch.no_grad():
        for features, target in pipe.cached_loader(model, loader):
            logits.append(model.get_classifier()(features)); labels.append(target)
    np.savez(RESULTS / f'probs_{dataset}_{name}_{split}.npz', logits=torch.cat(logits).numpy(), labels=torch.cat(labels).numpy(), classes=np.array(classes))
(RESULTS / f'exp1_row_{dataset}_{name}.json').write_text(json.dumps(row))
print('DONE', name)
