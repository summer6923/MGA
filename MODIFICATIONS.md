# Modification record

This distribution is a modified local Plato/KNOT research fork, not a verbatim
upstream checkout. Historical local modifications predate this release preparation;
their precise upstream commit ancestry is not established.

The corrected pipeline changes four pre-existing files:

- `knot_server.py`: request barrier integration and separation of target returns from retained-only recovery.
- `knot_client.py`: explicit phases, original-partition preservation, poisoning and per-dispatch execution evidence.
- `plato/clients/simple.py`: phase-aware reverse dispatch and propagation of training failures.
- `plato/trainers/basic.py`: explicit normalized reverse path, execution records and child-process exit checks.

Added implementation/support files include `mga_protocol.py`, `mga_data.py`,
`mga_reverse.py`, `mga_entry.py`, `run_mga.py`, `validate_mga_run.py`, regression
tests and the smoke configuration. Full behavior is described in README_MGA_FIX.md.

Publication preparation adds README/security/license/provenance documents, data
preparation and reference-evaluation commands, the exact full MNIST configuration,
packaging checks and sanitized evidence. Output locations in the full configuration
are relative; scientific values are unchanged. In `plato/config.py`, the inactive
legacy machine-specific default config location now points to the bundled smoke
configuration. All documented runs supply an explicit `--config`. A clean CPU-only
smoke test exposed a pre-existing hardcoded CUDA fallback in the same file; it now
falls back to CPU and honors explicit CPU selection. Available-GPU behavior remains
`cuda:0`. Three dedicated device-selection regressions accompany this correction.

Every copied Python file carries a prominent local-fork modification notice.
AST comparison verifies identical executable code for 212 of 213 copied Python
files. The remaining file differs only in the default configuration path and
CPU device-fallback constant; neither changes the validated GPU optimizer.
The scientific optimizer, datasets, sampler, scheduling, budgets and evaluation
operator were not modified during publication preparation.

Per-file input/release SHA-256 identities are in docs/source_manifest.json.
