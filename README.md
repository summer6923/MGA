<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:5B7CFA,50:8E2DE2,100:FF6EC4&height=230&section=header&text=MGA&fontSize=100&fontColor=ffffff&fontAlignY=32&desc=Mirror%20Gradient%20Ascent%20for%20Asynchronous%20Federated%20Unlearning&descSize=19&descAlignY=55" width="100%" alt="MGA — Mirror Gradient Ascent for Asynchronous Federated Unlearning"/>

### ✨ Forget Selectively · Learn Better · Together ✨

*A cleaner & fairer future for asynchronous federated learning.*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-2EA043?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)

<br>

![Federated Learning](https://img.shields.io/badge/Federated_Learning-8E2DE2?style=flat-square)
![Unlearning](https://img.shields.io/badge/Unlearning-00B8A9?style=flat-square)
![Async FL](https://img.shields.io/badge/Async_FL-F5A623?style=flat-square)

**[Why](#-why-mga) · [Method](#-method-overview) · [Results](#-results-snapshot) · [Quick Start](#-quick-start) · [Citation](#-citation)**

<br>

> **Staleness-aware reference modeling + constrained reverse optimization
> for efficient asynchronous federated unlearning.** 🪞

</div>

---

## 🪞 Why MGA?

| | |
|:---|:---|
| 🌱 **No affected-group retraining**<br>Unlearn target clients *without* retraining their whole group. | 🗄️ **Staleness-aware reference model**<br>Retained clients with different training speeds contribute with proper weights. |
| 🎯 **Adaptive reverse budget**<br>Each target client gets a personalized unlearning budget based on its own history. | 🚀 **Fast recovery with retained clients**<br>Aggregate returned models and quickly resume training. |

---

## 🧭 Method Overview

*From forgetting to a better global model, in four steps:*

| 1️⃣ Grouped Async FL | 2️⃣ Reference Model | 3️⃣ Reverse Optimization | 4️⃣ Resume Training |
|:---:|:---:|:---:|:---:|
| Clients are **grouped by local training speed**. | A **staleness-aware reference** model is built from retained clients. | Each target client runs **budget-bounded local reverse updates**. | Returned models are **aggregated** and training quickly resumes. |

> 📌 This page intentionally keeps things high-level — the full formulation lives in the paper,
> with implementation notes in [`docs/implementation.md`](docs/implementation.md).

---

## 📊 Results Snapshot

*Effective unlearning, stronger models:*

| 🧊 CIFAR-10 | ✍️ MNIST | 🛒 Purchase |
|:---:|:---:|:---:|
| **~25% faster** than full retraining to reach the 70% accuracy mark. | **Highest final accuracy** among no-retrain baselines. | **Backdoor robustness closest to retraining** in 15 / 18 settings. |

<sup>Full curves and the evaluation protocol are available in the paper and [`docs/`](docs).</sup>

---

## ⚡ Quick Start

```bash
# Linux/POSIX · Python 3.11
git clone https://github.com/summer6923/MGA.git
cd MGA

# Install the PyTorch wheels matching your CUDA first, then the runtime deps
python -m pip install -r requirements.txt

# Run a fresh MGA experiment (the output directory must be new)
python run_mga.py --config configs/mga_cifar10_full.yml \
    --data-path ./data --output ./runs/cifar10 --gpu 0 --timeout 86400

# Validate the run
python validate_mga_run.py --run ./runs/cifar10 --device cuda:0
```

> ⚠️ **Note** — the launcher requires **Linux/POSIX**; use `--gpu ""` for CPU-only runs.

### 📦 Ready-to-run settings

| Dataset | Configuration | Model |
|:---:|:---:|:---:|
| MNIST | `configs/mga_mnist_full.yml` | LeNet-5 |
| CIFAR-10 | `configs/mga_cifar10_full.yml` | ResNet-18 |
| Purchase | `configs/mga_purchase_full.yml` | Transformer |

### 🗂️ Repository Layout

```text
📦 MGA
├── 🧠 mga/                # core algorithm & asynchronous FL runtime
├── ⚙️  configs/           # ready-to-run YAML configs (MNIST · CIFAR-10 · Purchase)
├── 📚 docs/               # configuration & implementation notes
├── 🛠️  scripts/           # data download & evaluation helpers
├── 🚀 run_mga.py          # one-command experiment launcher
└── ✅ validate_mga_run.py # post-run validation & metrics
```

---

## 🔗 Useful Links

<div align="center">

[![Method Notes](https://img.shields.io/badge/Method_Notes-docs/implementation-4A90D9?style=for-the-badge&logo=readthedocs&logoColor=white)](docs/implementation.md)
[![Configurations](https://img.shields.io/badge/Configurations-docs/configurations-4CAF50?style=for-the-badge&logo=yaml&logoColor=white)](docs/configurations.md)
[![Built on](https://img.shields.io/badge/Built_on-Plato_/_KNOT-6A5ACD?style=for-the-badge&logo=github&logoColor=white)](NOTICE)

</div>

---

## 📚 Citation

If MGA contributes to your research, please consider citing:

```bibtex
@software{MGA_2026,
  title  = {MGA: Mirror Gradient Ascent for Asynchronous Federated Unlearning},
  author = {summer6923},
  year   = {2026},
  url    = {https://github.com/summer6923/MGA}
}
```

<sub>📖 The paper's BibTeX entry will be added upon publication.</sub>

---

<div align="center">

> ### 💬 *"Selective Forgetting for a More Trustworthy Future."*
> If you find MGA useful, please consider giving it a **⭐** — thanks for your interest! 🤖💙

**Open Research** ❤️ **Smarter AI** ❤️ **A More Trustworthy World**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:FF6EC4,50:8E2DE2,100:5B7CFA&height=140&section=footer" width="100%" alt="footer wave"/>

</div>
