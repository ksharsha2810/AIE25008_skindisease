"""Experiment 1: extended lightweight-backbone benchmark under the project's exact
frozen-feature linear-probe protocol (same splits, transforms, optimiser, early stopping).

Existing MobileNetV3-L / MobileNetV4-S / LCNet checkpoints are reused; the four extra
backbones are trained here. Also saves test-set probabilities for every run.
"""
import numpy as np
import torch

from common import DISEASE_DATASETS, RESULTS, pipeline, save

runtime, pipe = pipeline(extended=True)
frame = pipe.benchmark()
frame.to_csv(RESULTS / 'exp1_comparison_raw.csv', index=False)

probabilities = {}
for dataset in DISEASE_DATASETS:
    for name in runtime.MODELS:
        classes = sorted(pipe.splits(dataset)[0]['label'].unique())
        model = pipe.factory(name, len(classes), pretrained=False)
        model.load_state_dict(pipe.load_checkpoint(pipe.run / f'{dataset}_{name}.pt')['state'])
        loaders, classes = pipe.loaders(dataset, augment=False)
        model.eval()
        for split, loader in (('val', loaders[1]), ('test', loaders[2])):
            cached = pipe.cached_loader(model, loader)
            logits, labels = [], []
            with torch.no_grad():
                for features, target in cached:
                    logits.append(model.get_classifier()(features).cpu())
                    labels.append(target)
            np.savez(RESULTS / f'probs_{dataset}_{name}_{split}.npz', logits=torch.cat(logits).numpy(), labels=torch.cat(labels).numpy(), classes=np.array(classes))
save('exp1_done.json', {'models': list(runtime.MODELS)})
print('EXP1 DONE')
