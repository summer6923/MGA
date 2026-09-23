# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""Deterministic client-local poisoning and partition-preserving deletion.

The backing dataset is never modified. The rate is a fraction of unique examples
in a client's original training partition, NOT a fraction of the evaluation set.
Inversion happens in raw feature space, with the datasource normalization undone
and reapplied. The wrapper must be recreated when a virtual client ID changes.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, SubsetRandomSampler


NORMALIZATION = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "purchase": (None, None),
}


def indices_digest(indices):
    return hashlib.sha256(
        ",".join(str(int(i)) for i in sorted(set(indices))).encode("ascii")
    ).hexdigest()


def select_subset(indices, fraction, seed, client_id):
    """Stable membership without changing any process-global RNG state."""
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("The sample fraction must be finite and in [0, 1].")
    unique = sorted(set(int(i) for i in indices))
    ranked = sorted(unique, key=lambda i: hashlib.sha256(
        f"{int(seed)}:{int(client_id)}:{i}".encode("ascii")
    ).digest())
    return frozenset(ranked[:math.floor(fraction * len(unique))])


class IndexSampler:
    """Freeze actual sampler membership instead of rebuilding an IID partition."""
    def __init__(self, indices, seed):
        self.subset_indices = [int(i) for i in indices]
        if not self.subset_indices:
            raise ValueError("A training/unlearning partition must not be empty.")
        self.random_seed = int(seed)

    def get(self):
        generator = torch.Generator().manual_seed(self.random_seed)
        return SubsetRandomSampler(self.subset_indices, generator=generator)

    def num_samples(self):
        return len(self.subset_indices)


@dataclass(frozen=True)
class PoisonSpec:
    enabled: bool
    rate: float
    target_label: int
    seed: int
    dataset: str
    trigger: str = "invert"

    def __post_init__(self):
        name = self.dataset.lower().replace("-", "")
        object.__setattr__(self, "dataset", name)
        if name not in NORMALIZATION:
            raise ValueError(f"No validated trigger transform for dataset {name!r}.")
        if self.trigger != "invert":
            raise ValueError("Only the explicitly defined 'invert' trigger is supported.")
        if not math.isfinite(self.rate) or not 0 <= self.rate <= 1:
            raise ValueError("backdoor_poison_rate must be finite and in [0, 1].")
        classes = 100 if name == "purchase" else 10
        if isinstance(self.target_label, bool) or not isinstance(self.target_label, int):
            raise ValueError("backdoor_target_label must be an integer class ID.")
        if not 0 <= self.target_label < classes:
            raise ValueError("backdoor_target_label is outside the dataset class range.")


def invert_trigger(tensor, dataset):
    """Apply the same raw-space inversion in training AND evaluation."""
    name = dataset.lower().replace("-", "")
    if name not in NORMALIZATION:
        raise ValueError(f"Unsupported dataset {dataset!r}.")
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        raise TypeError("The trigger requires a floating-point, transformed tensor.")
    mean, std = NORMALIZATION[name]
    if mean is None:
        return 1.0 - tensor
    if tensor.ndim not in (3, 4) or tensor.shape[-3] != len(mean):
        raise ValueError("Unexpected normalized image shape for the trigger.")
    shape = (len(mean), 1, 1)
    mu = tensor.new_tensor(mean).reshape(shape)
    sigma = tensor.new_tensor(std).reshape(shape)
    return (1.0 - (tensor * sigma + mu) - mu) / sigma


class PoisonedDataset(Dataset):
    """Client-specific non-mutating view; evaluation never uses this wrapper."""
    def __init__(self, base, partition, spec, client_id, active):
        if isinstance(base, PoisonedDataset):
            raise ValueError("Do not stack poison wrappers across virtual clients.")
        self.base = base
        self.spec = spec
        self.client_id = int(client_id)
        self.partition = frozenset(int(i) for i in partition)
        if not self.partition or min(self.partition) < 0 or max(self.partition) >= len(base):
            raise ValueError("Invalid original training-partition indices.")
        self.poison_indices = select_subset(
            self.partition, spec.rate if spec.enabled and active else 0.0,
            spec.seed, client_id,
        )
        self.read_count = 0
        self.poison_read_count = 0
        self.classes = getattr(base, "classes", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        index = int(index)
        if index not in self.partition:
            raise IndexError("The sampler escaped the original client partition.")
        sample, label = self.base[index]
        self.read_count += 1
        if index in self.poison_indices:
            sample = invert_trigger(sample, self.spec.dataset)
            label = (torch.full_like(label, self.spec.target_label)
                     if torch.is_tensor(label) else self.spec.target_label)
            self.poison_read_count += 1
        return sample, label

    def manifest(self):
        return {
            "client_id": self.client_id,
            "unique_partition_size": len(self.partition),
            "poison_count": len(self.poison_indices),
            "partition_sha256": indices_digest(self.partition),
            "poison_indices_sha256": indices_digest(self.poison_indices),
            "configured_training_poison_rate": self.spec.rate,
            "target_label": self.spec.target_label,
            "dataset": self.spec.dataset,
            "trigger": self.spec.trigger,
            "seed": self.spec.seed,
        }
