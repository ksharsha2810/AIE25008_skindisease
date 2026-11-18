
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
import timm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

MODELS = ('mobilenetv3_small_100', 'mobilenetv2_100', 'efficientnet_b0', 'tf_efficientnet_lite0', 'ghostnet_100', 'mobilevit_xxs')
DISPLAY = ('MobileNetV3-Small', 'MobileNetV2', 'EfficientNet-B0', 'EfficientNet-Lite0', 'GhostNet-1.0', 'MobileViT-XXS')
TAXONOMY = {'akiec': 'actinic_keratosis_or_intraepithelial_carcinoma', 'bcc': 'basal_cell_carcinoma', 'bkl': 'benign_keratosis', 'df': 'dermatofibroma', 'mel': 'melanoma', 'nv': 'nevus', 'nev': 'nevus', 'vasc': 'vascular_lesion', 'ack': 'actinic_keratosis', 'scc': 'squamous_cell_carcinoma', 'sek': 'seborrheic_keratosis'}
SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}
GRADE_PATTERN = re.compile(r'(?:levle|level|grade|severity)[ _-]*([0-4])(?:[ _.-]|$)', re.I)


class DatasetSchemaError(ValueError):
    pass


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path, instruction):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f'Missing artifact {path}. {instruction}. If Colab deleted its disk, mount the same Drive root and rerun Setup.')
    return json.loads(path.read_text(encoding='utf-8'))


def grade_from_filename(path):
    match = GRADE_PATTERN.search(Path(path).name)
    return int(match.group(1)) if match else None


def metrics(truth, predicted, classes):
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, labels=list(range(len(classes))), average='macro', zero_division=0)
    return {'Precision': float(precision), 'Recall': float(recall), 'Accuracy': float(accuracy_score(truth, predicted)), 'F1': float(f1), 'Confusion': confusion_matrix(truth, predicted, labels=list(range(len(classes)))).tolist()}


class ImageDataset(Dataset):
    def __init__(self, frame, classes, size, training=False):
        self.frame = frame.reset_index(drop=True)
        self.lookup = {label: index for index, label in enumerate(classes)}
        operations = [transforms.Resize((size, size))]
        if training:
            operations += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
        operations += [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        self.transform = transforms.Compose(operations)
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row['path']) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, self.lookup[str(row['label'])]


class CBAM(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(), nn.Conv2d(hidden, channels, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, features):
        channel = torch.sigmoid(self.channel_mlp(features.mean((2, 3), keepdim=True)) + self.channel_mlp(features.amax((2, 3), keepdim=True)))
        features = features * channel
        spatial = torch.sigmoid(self.spatial(torch.cat([features.mean(1, keepdim=True), features.amax(1, keepdim=True)], dim=1)))
        return features * spatial


class DualHead(nn.Module):
    def __init__(self, backbone, channels, disease_classes, dropout):
        super().__init__()
        self.backbone = backbone
        self.cbam = CBAM(channels)
        self.dropout = nn.Dropout(dropout)
        self.disease_head = nn.Linear(channels, disease_classes)
        self.severity_head = nn.Linear(channels, 4)
    def forward_cached(self, features):
        if features.ndim != 4:
            raise RuntimeError(f'CBAM requires NCHW spatial features, received {tuple(features.shape)}')
        pooled = self.dropout(self.cbam(features).mean((2, 3)))
        return {'disease': self.disease_head(pooled), 'severity': self.severity_head(pooled)}
    def forward(self, images):
        return self.forward_cached(self.backbone.forward_features(images))


class TaskView(nn.Module):
    def __init__(self, model, task):
        super().__init__()
        self.model, self.task = model, task
    def forward(self, images):
        return self.model(images)[self.task]


class Pipeline:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cfg = read_json(self.root / 'config.json', 'Run Setup')
        self.run = self.root / ('demo' if self.cfg['demo'] else 'full')
        self.run.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cpu' if self.cfg['demo'] else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.seed()
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.signature = hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest()
        marker = self.run / 'run_config.json'
        if marker.exists() and read_json(marker, 'Run Setup')['signature'] != self.signature:
            raise RuntimeError(f'Configuration changed for {self.run}. Use a new SKIN_PIPELINE_ROOT or explicitly archive this mode directory before rerunning. Stale checkpoints are not reused.')
        write_json(marker, {'signature': self.signature, 'config': self.cfg, 'versions': {'torch': torch.__version__, 'timm': timm.__version__}})
    def seed(self):
        random.seed(self.cfg['seed'])
        np.random.seed(self.cfg['seed'])
        torch.manual_seed(self.cfg['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg['seed'])
    def preflight(self):
        if not os.environ.get('GEMINI_API_KEY'):
            raise RuntimeError('Missing GEMINI_API_KEY. Get a key at https://aistudio.google.com/apikey and put it in Colab Secrets or your environment; never paste it into notebook outputs.')
        needs_kaggle = any(not self.cfg['local_roots'].get(name) for name in self.cfg['slugs'])
        if needs_kaggle:
            credentials = Path(os.environ.get('KAGGLE_CONFIG_DIR', str(Path.home() / '.kaggle'))) / 'kaggle.json'
            env_auth = bool(os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'))
            if not credentials.exists() and not env_auth:
                raise RuntimeError('Missing Kaggle credentials: place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY. Obtain them at https://www.kaggle.com/settings/account (API). Alternatively set all three local_roots in Setup.')
        print('Credential presence checked (values are not printed). API authorization/quota is checked by the actual request.')
    def safe_extract(self, archive, destination):
        destination = Path(destination).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            broken = bundle.testzip()
            if broken:
                raise zipfile.BadZipFile(f'CRC failure: {broken}')
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination) or ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError(f'Unsafe archive member: {member.filename}')
            bundle.extractall(destination)
    def images(self, directory):
        return sorted(path for path in Path(directory).rglob('*') if path.suffix.lower() in SUFFIXES)
    def parse_disease(self, name, directory):
        required = ('image_id', 'dx') if name == 'HAM10000' else ('img_id', 'diagnostic')
        paths = self.images(directory)
        by_name, by_stem = {}, {}
        for path in paths:
            by_name.setdefault(path.name.lower(), []).append(path)
            by_stem.setdefault(path.stem.lower(), []).append(path)
        for csv in sorted(Path(directory).rglob('*.csv')):
            try:
                frame = pd.read_csv(csv)
            except (ValueError, UnicodeError, pd.errors.ParserError):
                continue
            frame.columns = [str(column).strip().lower() for column in frame.columns]
            if not set(required).issubset(frame.columns):
                continue
            group_column = 'lesion_id' if name == 'HAM10000' else 'patient_id'
            if group_column not in frame or frame[group_column].isna().any():
                raise ValueError(f'{csv}: missing {group_column}. Group-disjoint splitting needs lesion identifiers for HAM10000 and patient identifiers for PAD-UFES-20; use the original metadata mirror.')
            records, missing = [], []
            for _, row in frame.iterrows():
                identifier = str(row[required[0]]).replace('\\', '/').split('/')[-1].lower()
                candidates = by_name.get(identifier, by_stem.get(Path(identifier).stem, []))
                if not candidates:
                    missing.append(identifier)
                    continue
                label = str(row[required[1]]).strip().lower()
                if label not in TAXONOMY:
                    raise DatasetSchemaError(f'Unknown disease label {label!r} in {csv}; supply an explicit taxonomy mapping, never guess a column or class.')
                path = candidates[0]
                if len(candidates) > 1:
                    hashes = {hashlib.sha256(item.read_bytes()).hexdigest() for item in candidates}
                    if len(hashes) != 1:
                        raise ValueError(f'Ambiguous image identifier {identifier}: multiple different images.')
                records.append({'path': str(path.resolve()), 'label': TAXONOMY[label], 'group': str(row[group_column])})
            if missing:
                raise ValueError(f'{csv}: {len(missing)} metadata images missing, examples {missing[:3]}. Re-extract or redownload.')
            if not records:
                raise ValueError(f'No matching images for {csv}')
            return pd.DataFrame(records)
        raise ValueError(f'No compatible metadata CSV under {directory}. Required columns: {required if paths else ("image_id/dx or img_id/diagnostic")}; images found: {len(paths)}.')
    def parse_acne(self, directory):
        paths = self.images(directory)
        encoded_grades = {grade_from_filename(path) for path in paths} - {None}
        if 0 in encoded_grades and 4 in encoded_grades:
            raise DatasetSchemaError('Mixed zero-based and one-based ACNE04 filename grades; review labels before training.')
        filename_base = 0 if 0 in encoded_grades else 1
        print('ACNE04 filename grades:', sorted(encoded_grades), 'canonical mapping:', {grade: grade + (1 if filename_base == 0 else 0) for grade in sorted(encoded_grades)})
        fallback = {}
        for csv in sorted(Path(directory).rglob('*.csv')):
            try:
                frame = pd.read_csv(csv)
            except (ValueError, UnicodeError):
                continue
            frame.columns = [str(column).strip().lower() for column in frame.columns]
            image_column = next((column for column in ('filename', 'image', 'image_id', 'img_id', 'path') if column in frame), None)
            grade_column = next((column for column in ('grade', 'severity', 'level', 'label') if column in frame), None)
            if image_column and grade_column:
                for _, row in frame.iterrows():
                    fallback[Path(str(row[image_column])).stem] = row[grade_column]
        for annotation in sorted(Path(directory).rglob('*.txt')):
            for line in annotation.read_text(encoding='utf-8', errors='replace').splitlines():
                match = re.fullmatch(r'\s*(\S+\.(?:jpg|jpeg|png))\s*[,;\t ]+([0-4])\s*', line, re.I)
                if match:
                    fallback.setdefault(Path(match.group(1)).stem, match.group(2))
        records, unresolved = [], []
        for path in paths:
            grade = grade_from_filename(path)
            source = 'filename'
            if grade is not None and filename_base == 0:
                grade += 1
            if grade is None and path.stem in fallback:
                raw = fallback[path.stem]
                encoded = GRADE_PATTERN.search(str(raw) + '_')
                try:
                    grade = int(encoded.group(1)) if encoded else int(float(raw)) + (1 if self.cfg['severity_label_base'] == 0 else 0)
                    source = 'csv_or_txt'
                except (ValueError, TypeError):
                    grade = None
            if grade is None:
                for parent in path.parents:
                    if parent == Path(directory).parent:
                        break
                    match = GRADE_PATTERN.search(parent.name + '_')
                    if match:
                        grade, source = int(match.group(1)), 'folder'
                        break
                    if parent.name in ('0', '1', '2', '3', '4'):
                        grade = int(parent.name) + (1 if self.cfg['severity_label_base'] == 0 else 0)
                        source = 'folder'
                        break
            if grade not in (1, 2, 3, 4):
                unresolved.append(str(path))
                continue
            group = re.sub(r'(?i)(?:levle|level|grade|severity)[ _-]*[0-4][ _-]*', '', path.stem)
            sidecar = path.with_suffix('.txt')
            if source != 'filename' and sidecar.exists() and grade is None:
                raise ValueError('Unresolved sidecar grade; bounding-box counts are not automatically converted into clinical grades.')
            records.append({'path': str(path.resolve()), 'label': f'grade_{grade}', 'group': group, 'label_source': source})
        if unresolved or not records:
            raise DatasetSchemaError(f'ACNE04 unresolved severity labels for {len(unresolved)} images; examples: {unresolved[:3]}. Expected filename grades 0-3 or 1-4; fallback CSV/folder/filename-grade text only. Do not guess grades from lesion counts.')
        result = pd.DataFrame(records)
        if set(result['label']) != {'grade_1', 'grade_2', 'grade_3', 'grade_4'}:
            raise DatasetSchemaError('ACNE04 must contain all four grades; verify the mirror/extraction and severity_label_base.')
        print('ACNE04 label sources:', result['label_source'].value_counts().to_dict())
        return result
    def discover(self, name):
        local = self.cfg['local_roots'].get(name)
        directory = Path(local).expanduser().resolve() if local else self.root / 'data' / name / 'extracted'
        parser = self.parse_acne if name == 'ACNE04' else lambda location: self.parse_disease(name, location)
        last = None
        for attempt in range(3):
            try:
                if directory.exists():
                    try:
                        return parser(directory)
                    except DatasetSchemaError:
                        raise
                    except ValueError:
                        for nested in sorted(directory.rglob('*.zip')):
                            self.safe_extract(nested, nested.with_suffix(''))
                        return parser(directory)
                raise FileNotFoundError(f'Missing dataset path {directory}')
            except DatasetSchemaError as error:
                raise RuntimeError(f'{name}: label/schema validation failed: {error}. Existing downloads are preserved; redownloading cannot repair a label convention.') from error
            except (ValueError, FileNotFoundError, zipfile.BadZipFile, OSError) as error:
                last = error
                if local:
                    raise RuntimeError(f'Invalid local dataset {name} at {directory}: {error}. Set local_roots[{name!r}] to the complete extracted dataset or None for Kaggle download.') from error
                if attempt == 2:
                    break
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.config_dir = os.environ.get('KAGGLE_CONFIG_DIR', str(self.root.parent))
                api.config = str(Path(api.config_dir) / api.config_file)
                api.authenticate()
                download = directory.parent / 'archives'
                download.mkdir(parents=True, exist_ok=True)
                if attempt == 0 and list(download.glob('*.zip')):
                    try:
                        for archive in sorted(download.glob('*.zip')):
                            self.safe_extract(archive, directory)
                        continue
                    except (zipfile.BadZipFile, ValueError, OSError):
                        pass
                for archive in download.glob('*.zip'):
                    archive.unlink()
                try:
                    api.dataset_download_files(self.cfg['slugs'][name], path=str(download), force=True, quiet=False, unzip=False)
                    archives = sorted(download.glob('*.zip'))
                    if not archives:
                        raise FileNotFoundError('Kaggle returned no zip archive')
                    if directory.exists():
                        shutil.rmtree(directory)
                    for archive in archives:
                        self.safe_extract(archive, directory)
                except Exception as download_error:
                    last = download_error
                    print(f'{name}: download/extraction attempt {attempt + 1} failed ({type(download_error).__name__}). Retrying, without printing credentials.')
        raise RuntimeError(f'{name}: download/discovery failed after 3 attempts. Last error type: {type(last).__name__}. Check https://www.kaggle.com/datasets/{self.cfg["slugs"][name]}, accept terms, verify credentials, and set the current slug or local_roots in Setup.') from last
    def prepare(self, name):
        directory = self.run / 'manifests'
        directory.mkdir(exist_ok=True)
        manifest = directory / f'{name}.csv'
        if manifest.exists():
            frame = pd.read_csv(manifest, dtype=str)
            if not frame['path'].map(lambda value: Path(value).is_file()).all():
                raise RuntimeError(f'{manifest} references missing images. Restore data or delete only this manifest and its split files, then rerun dataset preparation.')
            return frame
        frame = self.discover(name)
        bad, hashes = [], []
        for index, path in enumerate(frame['path']):
            try:
                with Image.open(path) as image:
                    image.convert('RGB').load()
                hashes.append(hashlib.sha256(Path(path).read_bytes()).hexdigest())
            except (OSError, ValueError):
                bad.append(index)
                hashes.append(None)
        frame['sha256'] = hashes
        broken_paths = frame.iloc[bad]['path'].tolist()
        frame = frame.drop(index=bad).reset_index(drop=True)
        label_counts = frame.groupby('sha256')['label'].nunique()
        conflicting_hashes = label_counts[label_counts.gt(1)].index
        conflicting = frame[frame['sha256'].isin(conflicting_hashes)]
        write_json(directory / f'{name}_excluded.json', {
            'broken_paths': broken_paths,
            'conflicting_label_rows': conflicting[['path', 'label', 'group', 'sha256']].to_dict(orient='records'),
            'conflicting_label_groups': len(conflicting_hashes),
            'policy': 'Exclude every row of each identical-image group with conflicting labels; original files remain unchanged.',
        })
        frame = frame[~frame['sha256'].isin(conflicting_hashes)].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f'{name}: no unambiguous valid images remain after integrity filtering.')
        frame = frame.drop_duplicates('sha256').reset_index(drop=True)
        hash_groups = frame.groupby('sha256')['group'].nunique()
        if hash_groups.gt(1).any():
            raise ValueError(f'{name}: duplicated images span groups; reconcile group identifiers before splitting.')
        frame.to_csv(manifest, index=False)
        print(name, 'valid images:', len(frame), 'broken excluded:', len(bad), 'conflicting rows excluded:', len(conflicting), 'classes:', frame['label'].value_counts().to_dict())
        self.splits(name)
        return frame
    def splits(self, name):
        directory = self.run / 'manifests'
        paths = [directory / f'{name}_{part}.csv' for part in ('train', 'val', 'test')]
        if all(path.exists() for path in paths):
            frames = [pd.read_csv(path, dtype=str) for path in paths]
        else:
            manifest = directory / f'{name}.csv'
            if not manifest.exists():
                raise RuntimeError(f'Missing {manifest}. Run dataset preparation for {name}.')
            frame = pd.read_csv(manifest, dtype=str)
            if frame.groupby('label')['group'].nunique().min() < 5:
                raise ValueError(f'{name}: need at least five distinct groups per class for leakage-aware 60/20/20 splits.')
            splitter = StratifiedGroupKFold(5, shuffle=True, random_state=self.cfg['seed'])
            folds = [holdout for _, holdout in splitter.split(frame, frame['label'], frame['group'])]
            frames = [frame.iloc[np.concatenate(folds[2:])].copy(), frame.iloc[folds[1]].copy(), frame.iloc[folds[0]].copy()]
            classes = set(frame['label'])
            for part, subset in zip(('train', 'val', 'test'), frames):
                if set(subset['label']) != classes:
                    raise ValueError(f'{name} {part} is missing classes under group splitting. Use a reviewed grouped split; do not fall back to leaking images across sets.')
            if self.cfg['demo']:
                frames = [pd.concat([group.sample(min(len(group), self.cfg['samples_per_class'] if part == 0 else 1), random_state=self.cfg['seed']) for _, group in subset.groupby('label', sort=True)], ignore_index=True) for part, subset in enumerate(frames)]
            for path, subset in zip(paths, frames):
                subset.to_csv(path, index=False)
        for first in range(3):
            for second in range(first + 1, 3):
                if set(frames[first]['group']) & set(frames[second]['group']) or set(frames[first]['sha256']) & set(frames[second]['sha256']):
                    raise ValueError(f'{name}: group or identical-image leakage between splits.')
        return frames
    def loaders(self, name, augment=True):
        frames = self.splits(name)
        classes = sorted(set().union(*(set(frame['label']) for frame in frames)))
        for frame in frames:
            if not frame['path'].map(lambda value: Path(value).exists()).all():
                raise RuntimeError(f'{name}: split references missing files; restore dataset paths.')
        generator = torch.Generator().manual_seed(self.cfg['seed'])
        loaders = [DataLoader(ImageDataset(frame, classes, self.cfg['image_size'], augment and index == 0), batch_size=self.cfg['batch_size'], shuffle=index == 0, num_workers=0, generator=generator, drop_last=False) for index, frame in enumerate(frames)]
        return loaders, classes
    def factory(self, name, classes, pretrained=None):
        if name not in MODELS:
            raise ValueError(f'Unsupported model {name}; only the six registry models are allowed.')
        self.seed()
        if pretrained is None:
            pretrained = self.cfg['pretrained']
        try:
            model = timm.create_model(name, pretrained=pretrained, num_classes=classes, drop_rate=self.cfg['dropout'])
        except Exception as error:
            raise RuntimeError(f'Cannot build {name}. Required timm registry/pretrained weights unavailable. Install timm==1.0.15 and permit Hugging Face downloads for FULL mode; no silent random-weight fallback.') from error
        count = sum(parameter.numel() for parameter in model.parameters())
        if count >= 5_000_000:
            raise ValueError(f'{name} has {count:,} TOTAL parameters with {classes} classes; violates <5M cap.')
        return model.to(self.device)
    def evaluate(self, model, loader, classes, task=None):
        model.eval()
        truth, predicted = [], []
        with torch.no_grad():
            for images, labels in loader:
                result = model(images.to(self.device))
                logits = result[task] if task else result
                predicted.extend(logits.argmax(1).cpu().tolist())
                truth.extend(labels.tolist())
        return metrics(truth, predicted, classes)
    def latency(self, model, task=None):
        model.eval()
        images = torch.zeros(1, 3, self.cfg['image_size'], self.cfg['image_size'], device=self.device)
        def sync():
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        samples = []
        with torch.no_grad():
            for _ in range(3):
                model(images)
            for _ in range(self.cfg['latency_repeats']):
                sync()
                start = time.perf_counter()
                model(images)
                sync()
                samples.append((time.perf_counter() - start) * 1000)
        return float(np.median(samples))
    def checkpoint(self, path, model, metadata):
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        temporary = Path(str(path) + '.tmp')
        torch.save({'state': state, 'metadata': metadata, 'signature': self.signature}, temporary)
        temporary.replace(path)
    def load_checkpoint(self, path):
        if not Path(path).exists():
            raise RuntimeError(f'Missing checkpoint {path}. Run the corresponding training stage; results are never fabricated.')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['signature'] != self.signature:
            raise ValueError('Checkpoint belongs to a different configuration.')
        return checkpoint
    def fit(self, model, loaders, classes, path, task=None):
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.cfg['lr'], weight_decay=0.0001)
        best, stale, state = -1.0, 0, None
        cached = None
        if self.cfg['demo']:
            model.eval()
            cached = []
            with torch.no_grad():
                for loader in loaders[:2]:
                    features, labels = [], []
                    for images, target in loader:
                        spatial = model.backbone.forward_features(images.to(self.device)) if task else model.forward_features(images.to(self.device))
                        value = spatial if task else model.forward_head(spatial, pre_logits=True)
                        features.append(value.cpu())
                        labels.append(target)
                    cached.append(DataLoader(TensorDataset(torch.cat(features), torch.cat(labels)), batch_size=self.cfg['batch_size'], shuffle=len(cached) == 0))
        history = []
        for epoch in range(self.cfg['epochs']):
            model.train()
            if self.cfg['demo']:
                (model.backbone if task else model).eval()
                if task:
                    model.cbam.train(task == 'disease')
                    model.dropout.train()
            if task == 'severity':
                model.backbone.eval()
                model.cbam.eval()
            loss_sum, samples = 0.0, 0
            for images, labels in (cached[0] if cached else loaders[0]):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with (torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' and torch.cuda.is_bf16_supported() else nullcontext()):
                    if cached:
                        result = model.forward_cached(images) if task else model.get_classifier()(images)
                    else:
                        result = model(images)
                    logits = result[task] if task else result
                    loss = nn.functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(labels)
                samples += len(labels)
            validation = self.evaluate(model, loaders[1], classes, task)
            history.append({'epoch': epoch + 1, 'loss': loss_sum / samples, 'val_accuracy': validation['Accuracy']})
            print(f'{path.stem}: epoch {epoch + 1}/{self.cfg["epochs"]}, loss {loss_sum / samples:.4f}, val_accuracy {validation["Accuracy"]:.4f}')
            if validation['Accuracy'] > best:
                best, stale = validation['Accuracy'], 0
                state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            if not self.cfg['demo'] and stale >= 5:
                break
        if state is None:
            raise RuntimeError('No finite validation checkpoint was produced.')
        model.load_state_dict(state)
        self.checkpoint(path, model, {'classes': classes, 'task': task or 'disease', 'history': history})
        return best
    def benchmark(self):
        rows = []
        for dataset in ('HAM10000', 'PAD_UFES_20'):
            loaders, classes = self.loaders(dataset, augment=not self.cfg['demo'])
            for name in MODELS:
                self.seed()
                loaders, classes = self.loaders(dataset, augment=not self.cfg['demo'])
                model = self.factory(name, len(classes))
                path = self.run / f'{dataset}_{name}.pt'
                if path.exists():
                    model.load_state_dict(self.load_checkpoint(path)['state'])
                else:
                    if self.cfg['demo']:
                        for parameter in model.parameters():
                            parameter.requires_grad_(False)
                        for parameter in model.get_classifier().parameters():
                            parameter.requires_grad_(True)
                    self.fit(model, loaders, classes, path)
                validation = self.evaluate(model, loaders[1], classes)
                test = self.evaluate(model, loaders[2], classes)
                row = {'Dataset': dataset, 'Model': name, 'Name': DISPLAY[MODELS.index(name)], **{key: value for key, value in test.items() if key != 'Confusion'}, 'ValAccuracy': validation['Accuracy'], 'Params(M)': sum(parameter.numel() for parameter in model.parameters()) / 1e6, 'Inference(ms)': self.latency(model), 'Device': str(self.device), 'ImageSize': self.cfg['image_size'], 'Mode': 'DEMO_random_frozen_features' if self.cfg['demo'] else 'FULL_pretrained_finetuning'}
                rows.append(row)
                write_json(self.run / f'{dataset}_{name}_metrics.json', {'test': test, 'validation': validation, 'classes': classes})
                pd.DataFrame(rows).to_csv(self.run / 'comparison.csv', index=False)
                del model
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        return self.select()
    def select(self):
        path = self.run / 'comparison.csv'
        if not path.exists():
            raise RuntimeError('Missing comparison.csv. Run all six models on both datasets first.')
        frame = pd.read_csv(path)
        expected = {(dataset, model) for dataset in ('HAM10000', 'PAD_UFES_20') for model in MODELS}
        if set(zip(frame['Dataset'], frame['Model'])) != expected or len(frame) != 12:
            raise ValueError('Comparison is incomplete: all 12 model/dataset runs are required before selection.')
        aggregate = frame.groupby('Model')[['ValAccuracy', 'Params(M)', 'Inference(ms)']].mean().reindex(MODELS)
        def efficiency(series):
            spread = series.max() - series.min()
            return (series.max() - series) / spread if spread > 0 else pd.Series(1.0, index=series.index)
        aggregate['ParamsEfficiency'] = efficiency(aggregate['Params(M)'])
        aggregate['SpeedEfficiency'] = efficiency(aggregate['Inference(ms)'])
        aggregate['Score'] = 0.70 * aggregate['ValAccuracy'] + 0.15 * aggregate['ParamsEfficiency'] + 0.15 * aggregate['SpeedEfficiency']
        winner = aggregate.sort_values(['Score', 'ValAccuracy', 'Params(M)'], ascending=[False, False, True], kind='stable').index[0]
        write_json(self.run / 'selection.json', {'model': winner, 'dataset': self.cfg['primary_dataset'], 'score': float(aggregate.loc[winner, 'Score']), 'selection_split': 'validation', 'weights': [0.70, 0.15, 0.15]})
        aggregate.to_csv(self.run / 'selection_scores.csv')
        print(frame.drop(columns=['ValAccuracy']).to_string(index=False))
        print('\nValidation-based selection scores:\n', aggregate.to_string())
        print('Selected:', winner, 'source disease dataset:', self.cfg['primary_dataset'])
        return frame
    def new_dual(self, selection, classes):
        backbone = self.factory(selection['model'], len(classes), pretrained=False)
        backbone.eval()
        with torch.no_grad():
            features = backbone.forward_features(torch.zeros(1, 3, self.cfg['image_size'], self.cfg['image_size'], device=self.device))
        if features.ndim != 4:
            raise ValueError('Backbone does not expose NCHW fused features; add an explicit adapter, not a generic ViT token reshape.')
        model = DualHead(backbone, features.shape[1], len(classes), self.cfg['dropout']).to(self.device)
        count = sum(parameter.numel() for parameter in model.parameters())
        print('Dual-head wrapper total parameters (includes unused original classifier):', count)
        if count >= 5_000_000:
            raise ValueError(f'Wrapper has {count:,} parameters; <5M guard failed. Reduce CBAM channel bottleneck or explicitly revise the budget.')
        return model
    def disease_wrapper(self):
        selection = read_json(self.run / 'selection.json', 'Run benchmark and selection')
        loaders, classes = self.loaders(selection['dataset'], augment=not self.cfg['demo'])
        model = self.new_dual(selection, classes)
        path = self.run / 'dual_disease.pt'
        if path.exists():
            model.load_state_dict(self.load_checkpoint(path)['state'])
        else:
            base = self.load_checkpoint(self.run / f'{selection["dataset"]}_{selection["model"]}.pt')
            model.backbone.load_state_dict(base['state'])
            for parameter in model.severity_head.parameters():
                parameter.requires_grad_(False)
            for parameter in model.backbone.get_classifier().parameters():
                parameter.requires_grad_(False)
            if self.cfg['demo']:
                for parameter in model.backbone.parameters():
                    parameter.requires_grad_(False)
            self.fit(model, loaders, classes, path, 'disease')
        result = self.evaluate(model, loaders[2], classes, 'disease')
        write_json(self.run / 'dual_disease_metrics.json', result)
        print('Disease-trained CBAM wrapper test metrics:', result)
        return model
    def severity_transfer(self):
        selection = read_json(self.run / 'selection.json', 'Run benchmark')
        _, classes = self.loaders(selection['dataset'], augment=False)
        model = self.new_dual(selection, classes)
        disease = self.load_checkpoint(self.run / 'dual_disease.pt')
        model.load_state_dict(disease['state'])
        loaders, grades = self.loaders('ACNE04', augment=not self.cfg['demo'])
        path = self.run / 'dual_final.pt'
        if path.exists():
            model.load_state_dict(self.load_checkpoint(path)['state'])
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for parameter in model.severity_head.parameters():
                parameter.requires_grad_(True)
            self.fit(model, loaders, grades, path, 'severity')
        for key, value in disease['state'].items():
            if not key.startswith('severity_head.') and not torch.equal(value, model.state_dict()[key].cpu()):
                raise AssertionError(f'Severity transfer changed frozen disease/shared weights: {key}')
        result = self.evaluate(model, loaders[2], grades, 'severity')
        write_json(self.run / 'severity_metrics.json', result)
        print('ACNE04 severity test metrics:', result)
        disease_loaders, _ = self.loaders(selection['dataset'], augment=False)
        write_json(self.run / 'dual_final_disease_metrics.json', self.evaluate(model, disease_loaders[2], classes, 'disease'))
        return result
    def restore_final(self):
        selection = read_json(self.run / 'selection.json', 'Run benchmark')
        _, classes = self.loaders(selection['dataset'], augment=False)
        model = self.new_dual(selection, classes)
        model.load_state_dict(self.load_checkpoint(self.run / 'dual_final.pt')['state'])
        model.eval()
        return model, selection, classes
    def uncertainty(self, model, images):
        model.eval()
        original = [(module, module.training) for module in model.modules()]
        draws = {'disease': [], 'severity': []}
        try:
            model.dropout.train()
            with torch.no_grad():
                for _ in range(self.cfg['mc_passes']):
                    output = model(images)
                    for task in draws:
                        draws[task].append(output[task].softmax(1))
        finally:
            for module, training in original:
                module.training = training
        result = {}
        for task, samples in draws.items():
            probability = torch.stack(samples).mean(0)[0]
            entropy = float((-(probability * probability.clamp_min(1e-8).log()).sum() / math.log(len(probability))).clamp(0, 1))
            result[task] = {'index': int(probability.argmax()), 'confidence': float(probability.max()), 'normalized_entropy': entropy, 'probabilities': probability.cpu().tolist()}
        return result
    def cam(self, model, selection, images, index, rgb):
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        backbone = model.backbone
        if selection['model'] == 'mobilevit_xxs':
            layer = backbone.final_conv
            family = 'MobileViT hybrid: final convolution after token folding/fusion, NCHW; no CLS-token reshape'
        elif selection['model'] == 'ghostnet_100':
            layer = backbone.blocks[-1]
            family = 'GhostNet CNN: final spatial block before global pooling'
        elif selection['model'].startswith('mobilenetv3_'):
            layer = backbone.blocks[-1]
            family = 'MobileNetV3 CNN: final feature block before pooling; conv_head is outside forward_features'
        else:
            layer = backbone.conv_head
            family = 'MobileNet/EfficientNet CNN: final convolutional spatial features before pooling'
        captured = []
        handle = layer.register_forward_hook(lambda module, arguments, output: captured.append(tuple(output.shape)))
        try:
            with torch.no_grad():
                model(images)
        finally:
            handle.remove()
        if not captured or len(captured[0]) != 4 or min(captured[0][-2:]) < 2:
            raise RuntimeError(f'Grad-CAM target lacks a spatial NCHW grid: {captured}; use a pre-pooling architecture-specific layer.')
        model.eval()
        with torch.enable_grad(), GradCAM(model=TaskView(model, 'disease'), target_layers=[layer]) as explainer:
            heatmap = explainer(input_tensor=images.detach().requires_grad_(True), targets=[ClassifierOutputTarget(index)])[0]
        if not np.isfinite(heatmap).all() or float(heatmap.max() - heatmap.min()) <= 1e-8:
            raise RuntimeError('Grad-CAM is nonfinite or flat. Evidence is unavailable; no visual explanation is invented. Train a useful model or inspect the layer/gradients.')
        overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)
        Image.fromarray(overlay).save(self.run / 'gradcam_overlay.png')
        np.save(self.run / 'gradcam.npy', heatmap)
        np.save(self.run / 'gradcam_input.npy', rgb)
        peak = np.unravel_index(int(heatmap.argmax()), heatmap.shape)
        note = f'Disease-target Grad-CAM on {family}; peak at normalized row {peak[0] / heatmap.shape[0]:.3f}, column {peak[1] / heatmap.shape[1]:.3f}. Activation attribution is not a lesion mask or proof of clinical correctness.'
        return note
    def explain(self):
        model, selection, classes = self.restore_final()
        image_path = self.cfg['report_image']
        if image_path is None:
            image_path = self.splits(selection['dataset'])[2].iloc[0]['path']
        if not Path(image_path).exists():
            raise RuntimeError(f'Missing report_image {image_path}. Set report_image to a readable clinical image or None for a held-out dataset demonstration.')
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert('RGB').resize((self.cfg['image_size'], self.cfg['image_size'])), dtype=np.float32) / 255
        transform = ImageDataset(pd.DataFrame(), classes, self.cfg['image_size']).transform
        images = transform(Image.fromarray((rgb * 255).astype(np.uint8))).unsqueeze(0).to(self.device)
        self.seed()
        uncertainty = self.uncertainty(model, images)
        evidence = self.cam(model, selection, images, uncertainty['disease']['index'], rgb)
        disease = uncertainty['disease']
        severity = uncertainty['severity']
        disease_class = classes[disease.pop('index')]
        grade = severity.pop('index') + 1
        applicable = bool(self.cfg['acne_confirmed'] and self.cfg['report_image'])
        case_id = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()[:16]
        payload = {'schema_version': '1.0', 'case_id': case_id, 'mode': 'DEMO_NOT_CLINICAL' if self.cfg['demo'] else 'RESEARCH_NOT_CLINICAL', 'disease': {'class': disease_class, 'source_dataset': selection['dataset'], **disease}, 'severity': {'predicted_grade': grade, 'scale': 'ACNE04 acne grades 1-4', 'applicable': applicable, 'status': 'Acne confirmed externally; domain-shift limitations remain.' if applicable else 'Not clinically applicable: acne has not been externally confirmed. This grade must not be presented as general skin-disease severity.', **severity}, 'gradcam_evidence': evidence, 'specialist_review_required': bool(disease['normalized_entropy'] > 0.4 or severity['normalized_entropy'] > 0.4 or not applicable), 'review_reasons': []}
        if disease['normalized_entropy'] > 0.4:
            payload['review_reasons'].append('Disease normalized predictive entropy exceeds 0.4.')
        if severity['normalized_entropy'] > 0.4:
            payload['review_reasons'].append('Severity normalized predictive entropy exceeds 0.4.')
        if not applicable:
            payload['review_reasons'].append('ACNE04 severity is outside validated applicability for this image.')
        self.validate_payload(payload)
        write_json(self.run / 'payload.json', payload)
        print(json.dumps(payload, indent=2))
        return payload
    def validate_payload(self, payload):
        expected = {'schema_version', 'case_id', 'mode', 'disease', 'severity', 'gradcam_evidence', 'specialist_review_required', 'review_reasons'}
        if set(payload) != expected:
            raise ValueError('Payload schema contains unexpected or missing fields; no image/path field is permitted.')
        for task in ('disease', 'severity'):
            item = payload[task]
            for key in ('confidence', 'normalized_entropy'):
                if not 0 <= item[key] <= 1 or not math.isfinite(item[key]):
                    raise ValueError(f'Invalid {task} {key}')
            if not np.isclose(sum(item['probabilities']), 1, atol=1e-5):
                raise ValueError('Probabilities do not sum to one.')
        if payload['severity']['predicted_grade'] not in (1, 2, 3, 4):
            raise ValueError('Severity grade outside ACNE04 scale.')
        required = payload['disease']['normalized_entropy'] > 0.4 or payload['severity']['normalized_entropy'] > 0.4 or not payload['severity']['applicable']
        if payload['specialist_review_required'] != required:
            raise ValueError('Review flag contradicts uncertainty/applicability policy.')
        json.dumps(payload, allow_nan=False)
    def report_lines(self, payload):
        disease, severity = payload['disease'], payload['severity']
        return [f"Model disease class: {disease['class']}. This is a research prediction, not a clinician diagnosis.", f"Disease MC-dropout confidence: {disease['confidence']:.4f}; normalized predictive entropy: {disease['normalized_entropy']:.4f}.", f"ACNE04 head predicted acne grade: {severity['predicted_grade']}; applicability: {str(severity['applicable']).lower()}. {severity['status']}", f"Severity MC-dropout confidence: {severity['confidence']:.4f}; normalized predictive entropy: {severity['normalized_entropy']:.4f}.", payload['gradcam_evidence'], 'Specialist review required.' if payload['specialist_review_required'] else 'Uncertainty threshold does not trigger specialist review; clinical validation is still required.']
    def validate_report(self, report, payload):
        if set(report) != {'source_facts', 'sentences', 'specialist_review_required'}:
            raise ValueError('Gemini response must contain exactly source_facts, sentences, specialist_review_required.')
        if report['source_facts'] != payload or report['specialist_review_required'] != payload['specialist_review_required']:
            raise ValueError('Gemini altered source facts or the review flag; report rejected.')
        if report['sentences'] != self.report_lines(payload):
            raise ValueError('Gemini added, omitted, or changed narration. Only the verified sentence template is accepted; report rejected.')
        return {'faithfulness_passed': True, 'method': 'Exact source JSON equality, review-policy equality, closed sentence-template equality. This checks faithfulness, not medical correctness.'}
    def gemini(self):
        import google.generativeai as genai
        payload = read_json(self.run / 'payload.json', 'Run uncertainty and Grad-CAM')
        self.validate_payload(payload)
        key = os.environ.get('GEMINI_API_KEY')
        if not key:
            raise RuntimeError('Missing GEMINI_API_KEY; obtain at https://aistudio.google.com/apikey and set environment/Colab Secret.')
        genai.configure(api_key=key, transport='rest')
        policy = 'You are a fact-only formatter, never a diagnostician. You receive structured model facts only, never an image. Add no diagnosis, findings, lesion description, treatment, reassurance, or severity interpretation. Return JSON with exactly source_facts (copy the entire input verbatim), specialist_review_required (copy the input boolean), and sentences (the six exact strings in the following template instructions). If either normalized_entropy > 0.4, specialist review is mandatory; inapplicable acne severity also requires review. A non-applicable severity grade must never be described as clinically valid. Format confidence and entropy to four decimals. Template instructions: ' + json.dumps(self.report_lines(payload))
        model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=policy)
        last = None
        for attempt in range(3):
            try:
                response = model.generate_content(json.dumps(payload, allow_nan=False), generation_config={'temperature': 0, 'response_mime_type': 'application/json', 'max_output_tokens': 4096}, request_options={'timeout': 60, 'retry': None})
                report = json.loads(response.text)
                checks = self.validate_report(report, payload)
                write_json(self.run / 'report.json', report)
                write_json(self.run / 'faithfulness.json', checks)
                write_json(self.run / 'gemini_audit.json', {'model': 'gemini-2.5-flash', 'sdk': 'google-generativeai', 'prompt_sha256': hashlib.sha256(policy.encode()).hexdigest(), 'payload_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), 'attempts': attempt + 1, 'image_sent': False})
                print('\n'.join(report['sentences']))
                return report
            except Exception as error:
                last = error
                print(f'Gemini attempt {attempt + 1} failed ({type(error).__name__}); response not accepted.')
        raise RuntimeError('Gemini report unavailable or failed strict faithfulness validation after 3 attempts. Check API key, gemini-2.5-flash availability, region and free-tier quota at https://aistudio.google.com/ and https://ai.google.dev/gemini-api/docs/rate-limits. Legacy SDK may require migration; no local/fabricated report is substituted.') from last
    def expert_review(self):
        payload = read_json(self.run / 'payload.json', 'Build a real payload')
        report = read_json(self.run / 'report.json', 'Generate and validate Gemini report')
        checks = self.validate_report(report, payload)
        write_json(self.run / 'faithfulness.json', checks)
        path = self.run / 'expert_review.csv'
        columns = ['case_id', 'reviewer_id', 'faithful_to_source', 'uncertainty_clear', 'severity_scope_clear', 'clinically_safe_wording', 'overall_acceptable', 'comments']
        if not path.exists():
            pd.DataFrame([{column: payload['case_id'] if column == 'case_id' else '' for column in columns}]).to_csv(path, index=False)
        review = pd.read_csv(path, dtype=str).fillna('')
        completed = review[review['reviewer_id'].ne('') & review['overall_acceptable'].isin(['yes', 'no'])]
        print('Faithfulness:', checks)
        print(f'Expert review: {len(completed)} completed reviews; target {self.cfg["expert_sample_size"]} distinct cases. Blank rows are not evidence. Template: {path}')
        return checks
    def run_all(self):
        start = time.perf_counter()
        self.preflight()
        for dataset in ('HAM10000', 'PAD_UFES_20'):
            self.prepare(dataset)
        self.benchmark()
        self.disease_wrapper()
        self.prepare('ACNE04')
        self.severity_transfer()
        self.explain()
        self.gemini()
        self.expert_review()
        elapsed = time.perf_counter() - start
        print(f'End-to-end elapsed: {elapsed:.1f} seconds. No two-minute guarantee; downloads, model training, and API latency are real costs.')


MODELS = ('mobilenetv3_large_100.ra_in1k', 'mobilenetv4_conv_small.e2400_r224_in1k', 'lcnet_100.ra2_in1k')
DISPLAY = ('MobileNetV3-Large', 'MobileNetV4-Conv-Small', 'LCNet-1.0')
ACTIVE_TRANSFORM_MODEL = MODELS[0]
BaseImageDataset = ImageDataset
BasePipeline = Pipeline


class ImageDataset(BaseImageDataset):
    def __init__(self, frame, classes, size, training=False):
        super().__init__(frame, classes, size, training)
        pretrained_config = timm.get_pretrained_cfg(ACTIVE_TRANSFORM_MODEL)
        self.transform = timm.data.create_transform(input_size=(3, size, size), is_training=False, mean=pretrained_config.mean, std=pretrained_config.std, interpolation=pretrained_config.interpolation, crop_pct=1.0)


class CachedFeatures(Dataset):
    def __init__(self, features_path, labels_path):
        self.features = np.load(features_path, mmap_mode='r')
        self.labels = np.load(labels_path)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True)), int(self.labels[index])


class PretrainedStudy(BasePipeline):
    def __init__(self, root):
        super().__init__(root)
        global ACTIVE_TRANSFORM_MODEL
        selection = self.run / 'selection.json'
        ACTIVE_TRANSFORM_MODEL = read_json(selection, 'Run new-candidate comparison')['model'] if selection.exists() else MODELS[0]
        self.current_model_name = ACTIVE_TRANSFORM_MODEL
    def prepare(self, name):
        manifest = self.run / 'manifests' / f'{name}.csv'
        if not manifest.exists():
            raise RuntimeError(f'Missing {manifest}. Dataset downloading is disabled; restore the existing cleaned manifest.')
        frame = pd.read_csv(manifest, dtype=str)
        if not frame['path'].map(lambda value: Path(value).is_file()).all():
            raise RuntimeError(f'{name}: cached images missing. Restore them locally; no automatic redownload is allowed.')
        return frame
    def discover(self, name):
        raise RuntimeError(f'Dataset download disabled for {name}; use cached cleaned manifests and images.')
    def factory(self, name, classes, pretrained=None):
        global ACTIVE_TRANSFORM_MODEL
        ACTIVE_TRANSFORM_MODEL = self.current_model_name = name
        model = super().factory(name, classes, pretrained)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.get_classifier().parameters():
            parameter.requires_grad_(True)
        return model
    def cached_loader(self, model, loader, task=None, training=False):
        spatial = task is not None
        frame = loader.dataset.frame
        description = self.current_model_name + str(spatial) + frame[['path', 'label']].to_csv(index=False)
        fingerprint = hashlib.sha256(description.encode()).hexdigest()[:24]
        directory = self.run / 'features'
        directory.mkdir(exist_ok=True)
        features_path = directory / f'{fingerprint}.npy'
        labels_path = directory / f'{fingerprint}_labels.npy'
        marker = directory / f'{fingerprint}.json'
        if not (features_path.exists() and labels_path.exists() and marker.exists()):
            model.eval()
            ordered = DataLoader(loader.dataset, batch_size=self.cfg['batch_size'], shuffle=False, num_workers=0)
            temporary = directory / f'{fingerprint}.tmp.npy'
            mapped, targets, offset = None, [], 0
            with torch.no_grad():
                for images, labels in ordered:
                    backbone = model.backbone if spatial else model
                    features = backbone.forward_features(images.to(self.device))
                    if not spatial:
                        features = backbone.forward_head(features, pre_logits=True)
                    values = features.detach().cpu().numpy()
                    if mapped is None:
                        required_bytes = int(np.prod((len(frame), *values.shape[1:]))) * (2 if spatial else 4)
                        if shutil.disk_usage(directory).free < required_bytes + 200_000_000:
                            raise RuntimeError(f'Insufficient disk space under {directory}: need {required_bytes / 1e9:.2f} GB for this feature cache plus reserve.')
                        mapped = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16 if spatial else np.float32, shape=(len(frame), *values.shape[1:]))
                    mapped[offset:offset + len(values)] = values
                    targets.extend(labels.tolist())
                    offset += len(values)
                    print(f'Feature extraction {self.current_model_name}: {offset}/{len(frame)}', end='\r')
            if mapped is None or offset != len(frame):
                raise RuntimeError('Incomplete feature extraction; no partial cache is accepted.')
            mapped.flush()
            del mapped
            temporary.replace(features_path)
            np.save(labels_path, np.asarray(targets, dtype=np.int64))
            write_json(marker, {'model': self.current_model_name, 'samples': offset, 'spatial': spatial, 'pretrained': True})
            print()
        dataset = CachedFeatures(features_path, labels_path)
        return DataLoader(dataset, batch_size=self.cfg['batch_size'], shuffle=training, num_workers=0)
    def logits_cached(self, model, features, task=None):
        return model.forward_cached(features)[task] if task else model.get_classifier()(features)
    def evaluate(self, model, loader, classes, task=None):
        model.eval()
        cached = self.cached_loader(model, loader, task)
        truth, predicted = [], []
        with torch.no_grad():
            for features, labels in cached:
                logits = self.logits_cached(model, features.to(self.device), task)
                predicted.extend(logits.argmax(1).cpu().tolist())
                truth.extend(labels.tolist())
        return metrics(truth, predicted, classes)
    def fit(self, model, loaders, classes, path, task=None):
        self.seed()
        cached = self.cached_loader(model, loaders[0], task, training=True)
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=self.cfg['lr'], weight_decay=0.0001)
        epochs = self.cfg['disease_epochs'] if task == 'disease' else self.cfg['severity_epochs'] if task == 'severity' else self.cfg['epochs']
        best, stale, state, history = -1.0, 0, None, []
        for epoch in range(epochs):
            model.eval()
            if task:
                model.dropout.train()
                model.cbam.train(task == 'disease')
            loss_sum, samples = 0.0, 0
            for features, labels in cached:
                features, labels = features.to(self.device), labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.logits_cached(model, features, task)
                loss = nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss; checkpoint rejected.')
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(labels)
                samples += len(labels)
            validation = self.evaluate(model, loaders[1], classes, task)
            history.append({'epoch': epoch + 1, 'loss': loss_sum / samples, 'val_accuracy': validation['Accuracy'], 'val_macro_f1': validation['F1']})
            print(f'{path.stem}: epoch {epoch + 1}/{epochs}, loss {loss_sum / samples:.4f}, val_accuracy {validation["Accuracy"]:.4f}, val_macro_f1 {validation["F1"]:.4f}')
            if validation['Accuracy'] > best:
                best, stale = validation['Accuracy'], 0
                state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            if stale >= self.cfg['patience']:
                break
        if state is None:
            raise RuntimeError('No valid validation checkpoint.')
        model.load_state_dict(state)
        self.checkpoint(path, model, {'classes': classes, 'task': task or 'disease', 'history': history, 'protocol': self.cfg['protocol']})
        return best
    def benchmark(self):
        rows = []
        for dataset in ('HAM10000', 'PAD_UFES_20'):
            self.prepare(dataset)
            for name in MODELS:
                self.seed()
                path = self.run / f'{dataset}_{name}.pt'
                frame = self.splits(dataset)[0]
                classes = sorted(frame['label'].unique())
                model = self.factory(name, len(classes), pretrained=not path.exists())
                loaders, classes = self.loaders(dataset, augment=False)
                if path.exists():
                    model.load_state_dict(self.load_checkpoint(path)['state'])
                else:
                    self.fit(model, loaders, classes, path)
                validation = self.evaluate(model, loaders[1], classes)
                test = self.evaluate(model, loaders[2], classes)
                majority = loaders[0].dataset.frame['label'].value_counts().idxmax()
                majority_accuracy = float(loaders[2].dataset.frame['label'].eq(majority).mean())
                rows.append({'Dataset': dataset, 'Model': name, 'Name': DISPLAY[MODELS.index(name)], **{key: value for key, value in test.items() if key != 'Confusion'}, 'ValAccuracy': validation['Accuracy'], 'Params(M)': sum(parameter.numel() for parameter in model.parameters()) / 1e6, 'Inference(ms)': self.latency(model), 'MajorityBaselineAccuracy': majority_accuracy, 'Above70Percent': bool(test['Accuracy'] > self.cfg['accuracy_target']), 'Device': str(self.device), 'ImageSize': self.cfg['image_size'], 'Mode': self.cfg['protocol']})
                write_json(self.run / f'{dataset}_{name}_metrics.json', {'test': test, 'validation': validation, 'classes': classes, 'majority_baseline_accuracy': majority_accuracy})
                pd.DataFrame(rows).to_csv(self.run / 'comparison.csv', index=False)
                del model
        return self.select()
    def select(self):
        frame = pd.read_csv(self.run / 'comparison.csv')
        expected = {(dataset, name) for dataset in ('HAM10000', 'PAD_UFES_20') for name in MODELS}
        if len(frame) != len(expected) or set(zip(frame['Dataset'], frame['Model'])) != expected:
            raise RuntimeError('New-candidate comparison incomplete; all six new model/dataset runs are required.')
        aggregate = frame.groupby('Model')[['ValAccuracy', 'Params(M)', 'Inference(ms)']].mean().reindex(MODELS)
        def efficiency(series):
            spread = series.max() - series.min()
            return (series.max() - series) / spread if spread > 0 else pd.Series(1.0, index=series.index)
        aggregate['Score'] = 0.70 * aggregate['ValAccuracy'] + 0.15 * efficiency(aggregate['Params(M)']) + 0.15 * efficiency(aggregate['Inference(ms)'])
        winner = aggregate.sort_values(['Score', 'ValAccuracy', 'Params(M)'], ascending=[False, False, True], kind='stable').index[0]
        write_json(self.run / 'selection.json', {'model': winner, 'dataset': self.cfg['primary_dataset'], 'score': float(aggregate.loc[winner, 'Score']), 'selection_split': 'validation', 'weights': [0.70, 0.15, 0.15], 'protocol': self.cfg['protocol']})
        aggregate.to_csv(self.run / 'selection_scores.csv')
        print(frame.to_string(index=False))
        print('Selected new pretrained backbone:', winner)
        if not frame['Above70Percent'].all():
            print('Some candidates are below the 70% target. These are real held-out results, not a guarantee; consider end-to-end fine-tuning and calibration in a separately validated experiment.')
        return frame
    def disease_wrapper(self):
        selection = read_json(self.run / 'selection.json', 'Run new comparison')
        self.current_model_name = selection['model']
        return super().disease_wrapper()
    def severity_transfer(self):
        self.prepare('ACNE04')
        return super().severity_transfer()
    def cam(self, model, selection, images, index, rgb):
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        model.eval()
        backbone = model.backbone
        candidates = []
        if hasattr(backbone, 'final_conv'):
            candidates.append(backbone.final_conv)
        if hasattr(backbone, 'blocks'):
            candidates.append(backbone.blocks[-1])
        if hasattr(backbone, 'conv_head'):
            candidates.append(backbone.conv_head)
        layer = None
        for candidate in candidates:
            captured = []
            handle = candidate.register_forward_hook(lambda module, arguments, output: captured.append(tuple(output.shape)))
            try:
                with torch.no_grad():
                    model(images)
            finally:
                handle.remove()
            if captured and len(captured[0]) == 4 and min(captured[0][-2:]) >= 2:
                layer = candidate
                break
        if layer is None:
            raise RuntimeError('No executed pre-pooling spatial target layer for this backbone; evidence unavailable.')
        with torch.enable_grad(), GradCAM(model=TaskView(model, 'disease'), target_layers=[layer]) as explainer:
            heatmap = explainer(input_tensor=images.detach().requires_grad_(True), targets=[ClassifierOutputTarget(index)])[0]
        if not np.isfinite(heatmap).all() or float(heatmap.max() - heatmap.min()) <= 1e-8:
            raise RuntimeError('Grad-CAM is flat/nonfinite; no evidence note is fabricated.')
        Image.fromarray(show_cam_on_image(rgb, heatmap, use_rgb=True)).save(self.run / 'gradcam_overlay.png')
        np.save(self.run / 'gradcam.npy', heatmap)
        np.save(self.run / 'gradcam_input.npy', rgb)
        peak = np.unravel_index(int(heatmap.argmax()), heatmap.shape)
        return f'Disease-target Grad-CAM on {selection["model"]}, executed spatial feature layer {type(layer).__name__} before pooling; peak at normalized row {peak[0] / heatmap.shape[0]:.3f}, column {peak[1] / heatmap.shape[1]:.3f}. Attribution is not a lesion mask or proof of clinical correctness.'


Pipeline = PretrainedStudy
