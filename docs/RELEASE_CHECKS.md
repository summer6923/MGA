# Release preparation checks

## Completed scientific validation (before packaging)

The corrected source passed 25 regression tests, an eight-client asynchronous
execution smoke test, and one 100-client, 20-round MNIST experiment from random
initialization. The full run obtained 98.09% clean accuracy and 2.1884% ASR after
recovery. Its fixed-four reverse-step setting is preserved in configs/mga_mnist_full.yml.
These checks do not establish the adaptive-budget contribution or all paper results.

## Packaging validation

The release is built from the frozen source/results bundle identified by SHA-256
`108b2587e15ea5b74f875328c714968ffd51f1cf180ecd4fb98201399932ac09`.
Source AST equality was checked: 212 of 213 copied Python files are executable-code
identical. The remaining file replaces an obsolete absolute default config
path and fixes the CPU fallback in the device selector; available-GPU selection
is unchanged. Modification notices, public documentation, helper scripts and relative
output paths are added separately. Original live code and historical results are unchanged.

The parent Apache-2.0 LICENSE is included unchanged. Upstream notices and local
modifications are identified. Institutional/coauthor authority remains the
publishing maintainer's responsibility.

The pre-publication scan checks supported credential patterns and sensitive file
names without printing matched values. It also checks Python syntax, required files,
relative documentation links and copied-source/configuration hashes. See the
separate release-preparation report for the final count and outcome.

## Installation scope

The original full GPU experiment used an environment that inherited packages.
Release preparation additionally created a new **CPU-only virtual environment
with `include-system-site-packages = false`**, installed torch 2.13.0+cpu and
torchvision 0.28.0+cpu, then installed requirements.txt from the package index.
All installation commands succeeded, `pip check` found no broken requirements,
and **all 25 packaged regression tests passed**. These stages took 192.62 seconds.
Native summary and regression output are included as clean_install_result.json
and packaged_regressions.txt. requirements-cpu.lock.txt records 101 resolved
versions, excluding pip; no wheel-hash reproducibility claim is made.

This is a clean Python environment on the same Linux host, not a second-machine
or cross-platform test. The system interpreter and OS/driver remain host-provided.
The original full GPU benchmark was not retrained as part of publication preparation.
A first CPU-only smoke attempt stopped before training because the inherited
device selector defaulted to CUDA even when CUDA was unavailable. That portability
bug was corrected in the release copy and covered by three new tests. A fresh
second output directory then passed the full eight-client asynchronous smoke,
stage validation and reference-checkpoint evaluation. All 25 pipeline tests were
rerun after the correction, and all three device-selection tests passed.

The completed smoke dispatched four target clients, accepted every target return,
and recovered with retained clients only. It used only 200 samples per client;
its low final accuracy is not a paper benchmark. The full-run reference checkpoint
was separately reevaluated on CPU, reproducing 9,809/10,000 correct labels and
194/8,865 target responses. These inference counts do not represent new full training.
See packaged_runtime_result.json, portability_tests.txt and reference_inference.json.
No weights from the failed CPU attempt were reused.

## Publication status

Candidate version: 0.1.0. No GitHub repository, tag, public release or push has been
created by this preparation. The destination must be explicitly supplied and the
maintainer must have permission to publish the local modifications.
