# MGA — corrected research implementation

**v0.1.0 · MNIST validated · initial research release candidate**

This repository contains the corrected Mirror Gradient Ascent (MGA) implementation
built on a local Plato/KNOT fork. It fixes target-client exclusion before reverse
optimization and connects backdoor configuration to actual client-local training data.
It is **not a complete reproduction of every historical result in the paper**.

## Verified result

One fresh, seeded **100-client, 20-round MNIST** run obtained **98.09% clean test
accuracy** and **2.1884% ASR after retained-client recovery**. All ten target clients
executed reverse optimization; all returned target models were verified in the
aggregate before recovery. The run used **four fixed reverse steps per target**,
as explicitly set in the supplied full configuration. It is not an adaptive-budget ablation.

| Endpoint | Clean accuracy | ASR |
|---|---:|---:|
| Before request | 97.66% | 17.8342% |
| Reference | 96.89% | 16.2662% |
| After reverse optimization | 96.89% | 15.2284% |
| After retained-client recovery | **98.09%** | **2.1884%** |

Clean accuracy uses all 10,000 MNIST test images. ASR uses all 8,865 test images
whose original label is not the attack target, with raw-space inversion followed
by training-consistent normalization. This is one new corrected-code run, not the
older paper table's test-subsampling protocol. Results are observations, not promised
exact values on other hardware or independent seeds.

## Setup

Use **Linux/POSIX and Python 3.11**. Native Windows process execution is not supported
by the launcher; a Linux environment is required. Do not install another `plato`
package over the bundled `plato/` directory.

The following **CPU installation was tested in a new virtual environment with no
inherited site-packages**, followed by all 25 regression tests and `pip check`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu \
  torch==2.13.0 torchvision==0.28.0
python -m pip install -r requirements.txt
python scripts/download_mnist.py --data-path ./data
```

`requirements-cpu.lock.txt` records all 101 resolved runtime/build-package versions
from this clean installation (excluding pip); it is a version snapshot, not a
wheel-hash lock. The original full MNIST result used Python 3.11.15,
PyTorch 2.13.0+cu130 and an NVIDIA RTX 5000 Ada GPU. For GPU training, install a
compatible CUDA build of the pinned torch/torchvision versions instead of the CPU
wheels, then install `requirements.txt`. A fresh CUDA installation has not been
repeated as part of release preparation.

For CPU executions below, use `--gpu ""` with the launcher and `--device cpu`
with the evaluators. See [release checks](docs/RELEASE_CHECKS.md) for the precise
installation and runtime validation scope. Raw datasets and third-party package
binaries are not distributed here.

## Test the implementation

```bash
python tests/test_mga_pipeline.py
python tests/test_release_portability.py
python scripts/check_release.py --root . --self-test
```

The scientific implementation passed **25 regression tests**, including poisoning
membership and labels, last-feasible boundary handling, target scheduling despite
exclusion, stale/duplicate returns, small final target batches, worker reuse, and
failed subprocesses. Three additional device-selection tests passed, giving
**28 code tests in total**. The public package also adds file/notice/configuration
checks and six scanner self-tests. A complete eight-client asynchronous smoke run
and its stage validator passed in the clean CPU environment; reference-checkpoint
inference on CPU independently recovered 9,809 correct labels and 194 attack-target
responses. This does not add a second full-training benchmark.

## Run the exact validated full configuration from scratch

```bash
python run_mga.py --config configs/mga_mnist_full.yml \
  --data-path ./data --output ./runs/mnist-seed23 --gpu 0 --timeout 1800
python validate_mga_run.py --run ./runs/mnist-seed23 --device cuda:0
```

The output path must not already exist. The launcher seeds a new random model,
rejects external named checkpoints and pretrained initialization, and requires
all stages to complete. It does not use the distributed reference weights.

`configs/mga_mnist_full.yml` preserves the validated scientific settings: 100 clients,
five selected per event, request at round 10, ten targets, 20 rounds, training poison
rate 0.8, and fixed four-step reverse optimization. Only output locations were made
relative. The configured non-IID sampler is preserved; no train/test repartition was introduced.

## Small execution smoke check

```bash
python run_mga.py --config configs/mga_mnist_smoke.yml \
  --data-path ./data --output ./runs/smoke --gpu 0 --timeout 600
python validate_mga_run.py --run ./runs/smoke --device cuda:0
```

This eight-client test is deliberately too small for a meaningful benchmark accuracy.
It exercises asynchronous request handling and a final target batch smaller than the
normal aggregation threshold. A low ASR accompanied by poor accuracy is not evidence
of useful unlearning.

## Evaluate the distributed final checkpoint

Extract the separate weights/evidence archive next to this repository, then run:

```bash
python scripts/evaluate_reference.py \
  --checkpoint ../MGA-v0.1.0-evidence/mnist_seed23/models/lenet5.pth \
  --data-path ./data --device cuda:0
```

This checks saved-model inference only; it is separate from fresh training.

## Evidence and limits

[evidence/mnist_seed23](evidence/mnist_seed23/) contains the unchanged numeric
trajectory and stage results. `elapsed_time` is the simulator clock, not process
runtime. Raw checkpoints, sanitized logs and execution records are separate assets.

Full corrected CIFAR-10/Purchase training, multiple independent seeds, matched
retained-only retraining/stop-participation controls and historical timing claims
are not validated by this release. Successful execution is not certified removal
of historical influence. Mid-request resume is explicitly unsupported.

Detailed implementation choices: [README_MGA_FIX.md](README_MGA_FIX.md).
Licensing and attribution: [LICENSE](LICENSE), [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Security scope: [SECURITY.md](SECURITY.md).
