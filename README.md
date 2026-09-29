<div align="center">

<img src="docs/assets/hero.png?v=3" width="100%" alt="MGA — Mirror Gradient Ascent for Asynchronous Federated Unlearning"/>

### ✨ Forget Selectively · Learn Better · Together ✨

> **Staleness-aware reference modeling + constrained reverse optimization
> for efficient asynchronous federated unlearning.** 🪞

</div>

---

<div align="center">
<img src="docs/assets/sec-why.png?v=3" height="58" alt="Why MGA?"/>
</div>

<div align="center">

<img src="docs/assets/card-why1.png?v=3" width="49%" alt="No affected-group retraining"/> <img src="docs/assets/card-why2.png?v=3" width="49%" alt="Staleness-aware reference model"/>

<img src="docs/assets/card-why3.png?v=3" width="49%" alt="Adaptive reverse budget"/> <img src="docs/assets/card-why4.png?v=3" width="49%" alt="Fast recovery with retained clients"/>

</div>

---

<div align="center">
<img src="docs/assets/sec-method.png?v=3" height="58" alt="Method Overview"/>
</div>

<div align="center">

<img src="docs/assets/card-m1.png?v=3" width="24%" alt="Step 1 — Grouped Async FL"/> <img src="docs/assets/card-m2.png?v=3" width="24%" alt="Step 2 — Reference Model"/> <img src="docs/assets/card-m3.png?v=3" width="24%" alt="Step 3 — Reverse Optimization"/> <img src="docs/assets/card-m4.png?v=3" width="24%" alt="Step 4 — Resume Training"/>

</div>

> 📌 This page intentionally keeps things high-level — the full formulation lives in the paper,
> with implementation notes in [`docs/implementation.md`](docs/implementation.md).

---

<div align="center">
<img src="docs/assets/sec-results.png?v=3" height="58" alt="Results Snapshot"/>
</div>

<div align="center">

<img src="docs/assets/card-res1.png?v=3" width="32%" alt="CIFAR-10 results"/> <img src="docs/assets/card-res2.png?v=3" width="32%" alt="MNIST results"/> <img src="docs/assets/card-res3.png?v=3" width="32%" alt="Purchase results"/>

</div>

<sup>Illustrative sketches — full curves and the evaluation protocol are available in the paper and [`docs/`](docs).</sup>

---

<div align="center">
<img src="docs/assets/sec-quickstart.png?v=3" height="58" alt="Quick Start"/>
</div>

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

<div align="center">
<img src="docs/assets/sec-links.png?v=3" height="58" alt="Useful Links"/>
</div>

<div align="center">

<a href="docs/implementation.md"><img src="docs/assets/btn-method.png?v=3" height="48" alt="Method Notes"/></a> <a href="docs/configurations.md"><img src="docs/assets/btn-configs.png?v=3" height="48" alt="Configurations"/></a> <a href="NOTICE"><img src="docs/assets/btn-plato.png?v=3" height="48" alt="Built on Plato / KNOT"/></a>

</div>

---

<div align="center">
<img src="docs/assets/sec-citation.png?v=3" height="58" alt="Citation"/>
</div>

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

<img src="docs/assets/footer.png?v=3" width="72%" alt="Open Research · Smarter AI · A More Trustworthy World"/>

</div>
