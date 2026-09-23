# Publishing this candidate

This directory is the source-only repository tree. Do not publish the old local
MGA reconstruction directory or the original audit/backup workspace instead.

Before public upload, the maintainer must confirm the intended GitHub repository
and authority to publish the local modifications under the included license.
No repository, commit, tag or release has been created by preparation.

Run `python scripts/check_release.py --root . --self-test` on the exact source
tree that will be uploaded. Inspect the actual staged Git changes; do not include
private environments, credentials, datasets, generated runs or old project history.
The standalone weights/evidence ZIP belongs in a release asset, not the code tree.
Its MANIFEST.json identifies sanitized text versus unchanged numeric and weight files.

Suggested public title: **MGA v0.1.0 — MNIST-validated research implementation**.
Suggested description: Corrected client-local poisoning and explicit asynchronous
unlearning request handling, with one validated 100-client MNIST configuration.
Do not describe this release as a reproduction of all paper tables or a certified
unlearning system.

Preserve LICENSE, NOTICE, THIRD_PARTY_NOTICES.md and modification notices. Retain
source_manifest.json and the reference result hashes when mirroring the release.
Choose an immutable source tag only after checking that the destination has no
conflicting version. Existing tags or releases must not be silently replaced.
