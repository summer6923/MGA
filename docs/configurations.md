# Dataset configurations

The README is intentionally blank. These notes document the supplied YAML files.

| Dataset | Full configuration | Model | Clients / selected / minimum aggregation | Targets | Rounds |
| --- | --- | --- | --- | --- | --- |
| MNIST | `configs/mga_mnist_full.yml` | LeNet-5 | 100 / 5 / 5 | 10 | 20 |
| CIFAR-10 | `configs/mga_cifar10_full.yml` | ResNet-18 | 100 / 30 / 15 | 10 | 70 |
| Purchase | `configs/mga_purchase_full.yml` | Transformer | 50 / 30 / 15 | 5 | 120 |

The existing MNIST YAML is unchanged. CIFAR-10 and Purchase are adapted from the
archived `cifar10_ours_w10_C_struct_ret6_s1.yml` and `purchase_ours_BD_P_w7_rec8.yml`.
Their original client counts, request round, non-IID sampler, training and recovery
budgets, optimizer, learning-rate schedule, and seed are preserved. The explicit
corrected MGA path uses client-level removal, inversion poisoning, radius 0.5, and
scalar reverse learning rates 0.0009 and 0.000525, respectively. The radius is a
corrected-path starting setting, not a value validated by historical benchmarks.
Full corrected CIFAR-10/Purchase benchmark accuracies have NOT been reproduced by
adding these files. Both new full configs use adaptive budgets bounded from 1 to 2;
the validated MNIST config retains its fixed four-step setting.

## Run

Use Linux/POSIX and Python 3.11. Install compatible CPU/CUDA PyTorch wheels, then
`python -m pip install -r requirements.txt`. No separate Plato installation is needed.
The optional legacy optimized-clustering dependencies are in
`requirements-optional.txt`; the supplied YAMLs do not require them.

```bash
python run_mga.py --config configs/mga_cifar10_full.yml \
  --data-path ./data --output ./runs/cifar10-seed1 --gpu 0 --timeout 86400
python validate_mga_run.py --run ./runs/cifar10-seed1 --device cuda:0

python run_mga.py --config configs/mga_purchase_full.yml \
  --data-path ./data --output ./runs/purchase-seed50 --gpu 0 --timeout 86400
python validate_mga_run.py --run ./runs/purchase-seed50 --device cuda:0
```

The output directory must be new. `--timeout` limits the actual process lifetime,
not the simulator clock. For CPU use `--gpu ""` and `--device cpu`.

## Data and evaluation

MNIST and CIFAR-10 retain their torchvision training/test splits. CIFAR-10 uses
random horizontal flips and crops during training, with the existing normalization
(mean 0.485/0.456/0.406, standard deviation 0.229/0.224/0.225). Dataset caches should
be prepared before starting concurrent clients.

Purchase accepts `purchase_numpy.npz` with arrays `X` (600 features) and `Y`
(zero-based labels 0–99). The original seed-0 shuffle and first 20,000 training /
next 20,000 test examples are preserved. The loader can also prepare the cache
from the original `dataset_purchase` file. There is no new 80/20 split. The legacy
name `multilayer` selects the bundled Transformer: width 512, eight heads, six layers,
100 output classes. Its checkpoint structure is retained for compatibility.

`validate_mga_run.py` supports all three datasets. It checks completion, target
returns, absence of targets in recovery, actual poison reads, and reconstruction
of the returned-model aggregate before reporting stage metrics. ASR uses all
non-target test examples and the training-consistent inversion convention.
No expected accuracy, historical checkpoint, or bundled evidence file is required.
