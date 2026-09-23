# Changelog

## 0.1.0 — initial MNIST-validated research candidate

- Fix target exclusion before reverse optimization with a request-time barrier and explicit target-return aggregation.
- Activate deterministic client-local poisoning using the configured rate and target label.
- Preserve original client sampler membership and prevent stale worker settings from leaking between clients.
- Use a documented normalized-ascent reverse path with last-feasible rejection; fail on incomplete executions.
- Include 25 passing regression tests, an asynchronous smoke configuration and a completed 100-client MNIST run.
- Include the full fixed-four-step MNIST configuration, source/result hashes, license notices, setup and release checks.
- Fix the inherited device selector's CUDA-only fallback; add three CPU/GPU-selection regression tests.
- Validate installation in a fresh CPU-only Python environment without inherited site-packages.

This release does not claim a reproduced three-dataset benchmark, independent-seed
robustness, certified unlearning, or reproduction of the historical timing/ASR table.
Public upload and a repository URL are pending; no GitHub release has been created.
