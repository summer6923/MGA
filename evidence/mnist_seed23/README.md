# Corrected MNIST reference run

This is one completed run, not a statistical average and not the historical paper's ASR protocol.
Clean test set: 10,000 examples. ASR denominator: 8,865 non-target-label examples.
Trigger: raw-space inversion followed by the same normalization as training.

`trajectory.csv` preserves the 20 original numerical rows. `elapsed_time` is the
simulator clock; it is NOT process runtime. Actual observed process time was
677.0977538260631 seconds. The simulator ended at 163.05871200561523 seconds.

Ten target clients returned models, all four accepted steps each; 9,600 poisoned
training-example accesses were observed (including repeated epochs), with zero
target-client recovery dispatches. The full config keeps the historical fixed-four
budget override: this result does NOT demonstrate the adaptive-budget benefit.

Checkpoints and sanitized execution logs are distributed in the separate weights
and evidence archive. Training commands never load those checkpoints.
