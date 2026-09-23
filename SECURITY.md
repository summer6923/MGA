# Security scope

This is research software for a trusted local or isolated experimental environment,
not a hardened federated-learning service for mutually untrusted participants.
The inherited transport uses Python serialization. Do not expose it directly to
untrusted networks and do not accept payloads or checkpoints from untrusted peers.
The supplied launcher binds to localhost and uses a fresh output directory.

No credentials, private keys, raw private datasets, model-service tokens or unrelated
project history are needed. Keep generated data, runs, environments and credential
files out of Git. The repository's .gitignore and scripts/check_release.py support
that check but do not replace review of the actual final Git diff and release files.

Reference-model evaluation uses torch.load(weights_only=True). The research
training transport still has a different trust model; this is not a security audit
of all inherited Plato code. Pattern scanning cannot guarantee the absence of
all secrets or discover every logic vulnerability.

Report a security issue privately to the repository maintainer once a destination
repository and contact are configured. Do not place credentials or sensitive data
in public issues. No support or deployment guarantee is implied.
