# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""Evaluation shared by the MNIST, CIFAR-10 and Purchase stage validator.

Uses the bundled test transforms and the historical Purchase 20k/20k split.
No data are sampled according to training poisoning rate during ASR evaluation.
"""
from pathlib import Path
import hashlib
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from mga.data import invert_trigger


def dataset_name(config: dict[str, Any]) -> str:
    name = config['data']['datasource'].lower().replace('-', '')
    if name not in {'mnist', 'cifar10', 'purchase'}:
        raise ValueError(f'Unsupported evaluation dataset: {name}')
    return name


def checkpoint_name(config: dict[str, Any]) -> str:
    name = config['trainer']['model_name']
    if name not in {'lenet5', 'resnet_18', 'multilayer'}:
        raise ValueError(f'Unsupported MGA checkpoint model: {name}')
    return name


def create_model(config: dict[str, Any]) -> torch.nn.Module:
    name = dataset_name(config)
    params = dict(config.get('parameters', {}).get('model', {}))
    model_name = checkpoint_name(config)
    if name == 'mnist' and model_name == 'lenet5':
        from plato.models.lenet5 import Model
        params.setdefault('num_classes', 10)
        return Model(**params)
    if name == 'cifar10' and model_name == 'resnet_18':
        from plato.models.resnet import Model
        params.setdefault('num_classes', 10)
        return Model.get(model_name=model_name, **params)
    if name == 'purchase' and model_name == 'multilayer':
        from plato.models.multilayer import Model
        params.setdefault('num_classes', 100)
        return Model(**params)
    raise ValueError(f'Dataset/model mismatch: {name}/{model_name}')


def load_test_dataset(config: dict[str, Any]) -> Dataset:
    name = dataset_name(config)
    root = Path(config['data']['data_path']).expanduser().resolve()
    if name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
        return datasets.MNIST(str(root), train=False, download=False, transform=transform)
    if name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(
                [.485, .456, .406], [.229, .224, .225])])
        return datasets.CIFAR10(str(root), train=False, download=False, transform=transform)
    cache = root / 'purchase_numpy.npz'
    if not cache.is_file():
        raise FileNotFoundError(f'Prepare Purchase cache before evaluation: {cache}')
    from plato.datasources.purchase import DataSource
    # Reuse the actual loader's seed-0 shuffle, first 20k train, next 20k test.
    # Avoid __init__ here: evaluation must never initiate a download.
    _, test = DataSource.extract_data(DataSource.__new__(DataSource), str(root))
    if len(test) != 20000:
        raise ValueError('Purchase evaluation requires the historical 20,000-row test split.')
    return test


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def evaluate_checkpoint(state_path: Path, config: dict[str, Any],
                        dataset: Dataset, device: str, batch_size: int = 256) -> dict:
    if batch_size < 1:
        raise ValueError('Evaluation batch size must be positive.')
    name = dataset_name(config)
    target = int(config['clients'].get('backdoor_target_label', 1))
    classes = 100 if name == 'purchase' else 10
    if not 0 <= target < classes:
        raise ValueError('Attack target is outside the dataset label range.')
    model = create_model(config)
    model.load_state_dict(torch.load(state_path, map_location='cpu', weights_only=True), strict=True)
    model.to(device).eval()
    correct = total = hits = eligible_count = 0
    for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        images, labels = images.to(device), labels.to(device)
        correct += int((model(images).argmax(1) == labels).sum())
        total += len(labels)
        eligible = labels != target
        if bool(eligible.any()):
            triggered = invert_trigger(images[eligible], name)
            hits += int((model(triggered).argmax(1) == target).sum())
            eligible_count += int(eligible.sum())
    if total == 0 or eligible_count == 0:
        raise ValueError('Evaluation requires nonempty clean and non-target test sets.')
    return {'clean_correct': correct, 'clean_total': total,
            'clean_accuracy': correct / total, 'asr_hits': hits,
            'asr_total': eligible_count, 'asr': hits / eligible_count,
            'checkpoint_sha256': file_sha256(state_path)}
