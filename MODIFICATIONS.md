# Modification record

This is a modified local Plato/KNOT research fork, not an official upstream release.
Its original upstream commit ancestry is not established. License and attribution
notices are retained in LICENSE, NOTICE and THIRD_PARTY_NOTICES.md.

The corrected implementation separates target dispatch from retained-client recovery,
connects poisoning settings to client-local data, preserves sampler membership, and
uses explicit normalized ascent with last-feasible rejection. Worker state resets,
failed-subprocess checks and CPU device fallback are also local changes.

The repository cleanup groups the application into `mga/`. Shared rollback,
forget/retain sampling and optimized-grouping support live in `mga/legacy/`.
`plato/` retains the runtime and supported MNIST, CIFAR-10 and Purchase components;
unrelated framework integrations and unused backup modules have been removed.

Stored experiment evidence and obsolete release bookkeeping have been removed
from the working tree, not from Git history. The README is intentionally empty
pending a separate rewrite. Runtime execution records and evaluation remain enabled.
Training configurations and the numerical reverse-optimization implementation
are preserved; entrypoint imports and package paths are updated.

A second cleanup removes the public test suite after archiving it externally and
trims unused sampler, compression, and third-party optimizer/scheduler integrations.
Core transport, training, model definitions, and IID/non-IID partition membership
remain. Legacy grouping solvers are lazy imports with optional requirements.
The NumPy scheduler product call is updated for the pinned NumPy version. Purchase
can use its existing NPZ cache without redownloading; its original split remains.
Stage evaluation now supports all three configured datasets. New CIFAR-10/Purchase
YAMLs adapt archived training settings to the corrected path and do not claim that
their full historical benchmarks were reproduced. The README remains empty.

Protocol details remain in [docs/implementation.md](docs/implementation.md).
