# Third-party notices and licensing review

## Source code

The locally installed parent Plato source includes Apache License 2.0. Its exact
license text has been restored to this package; SHA-256:
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
The earlier code/results ZIP omitted a top-level license, so it should not be
published as a standalone licensed distribution without this repair.

The upstream TL-System/plato repository identifies Apache-2.0 licensing, and the
KNOT authors identify their implementation as part of the Plato source project.
This is attribution/provenance support, not a claim that every local file is
byte-identical to a specific upstream commit: the local historical fork has no
recoverable exact Git revision in this package.

- Plato: https://github.com/TL-System/plato
- KNOT, Ningxin Su and Baochun Li, IEEE INFOCOM 2023:
  https://ningxinsu.github.io/projects/infocom23/
- Apache License redistribution conditions:
  https://www.apache.org/licenses/LICENSE-2.0

Source notices were inspected for conflicting per-file license declarations;
none were detected by the recorded text scan. Modification notices have been
added to copied Python files without removing existing contents or attributions.
The original parent NOTICE file was absent, as recorded in the preparation check.

## Dependencies, datasets and checkpoints

Python dependencies remain separate packages governed by their respective licenses.
This archive does not bundle their wheels, CUDA runtime binaries, MOSEK license
files, raw MNIST/CIFAR-10/Purchase data or private datasets. The runtime retains
shared grouping and transport support in addition to the selected MNIST path.
Optimized-clustering solver usage is outside the validated configuration; consult
the relevant vendor terms before enabling it.

The separate reference checkpoints were produced by the corrected MNIST experiment.
They are research artifacts, not third-party pretrained weights. Never execute or
unpickle arbitrary downloaded objects; use the supplied weights-only evaluator.

## Maintainer confirmation before public upload

The observed Apache license permits redistribution subject to its conditions;
this preparation preserves the license and notices. It cannot establish authorship,
employment agreements, institutional ownership, or coauthor consent for local
modifications. The publishing maintainer must have authority to release those
additions under the accompanying license. No independent legal clearance is claimed.
