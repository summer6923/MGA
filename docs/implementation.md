# Corrected MGA execution path

The `mga/` package corrects two defects in the inherited KNOT/Plato path:
target exclusion previously prevented target-local reverse optimization, and the
backdoor YAML switches were not connected to the training data. Existing results,
checkpoints and manuscripts are not migrated or relabelled as corrected results.

## Run a fresh execution check

Run from the repository root with the dependencies in `requirements.txt`.
The commands below are Linux/POSIX commands. The launcher requires a NEW output
path and cached datasets (or the underlying datasource's documented download).

```bash
python tests/test_mga_pipeline.py
python run_mga.py --config configs/mga_mnist_smoke.yml \
  --data-path /path/to/data --output /new/path/mga-smoke --gpu 0
python validate_mga_run.py --run /new/path/mga-smoke --device cuda:0
```

The eight-client smoke configuration is for software execution validation, not a
paper benchmark. Run duration and sample counts must never be substituted into
the paper. This implementation note describes the protocol, not a guarantee of
benchmark performance on other machines.

## Scheduling fix

The corrected path is active with `trainer.mga_enabled: true` or strategy `mga`.
For compatibility it also activates for the historical strategy `knot` combined
with `skip_cluster_rollback_after_deletion: true`. Explicit `mga_enabled: false`
disables it; scratch/KNOT-retrain flags leave the retraining path distinct.

At the request the server freezes accepted retained-client records, drains and
rejects old in-flight/buffered updates, and constructs one common reference. It
schedules EVERY target, in batches if there are more targets than selection slots.
All returns carry the request ID, dispatch ID, and fresh child-process execution
evidence. No retained-only recovery aggregation can discard these target outputs.
Recovery begins only after all validated returns have been aggregated. Target
clients are then excluded from all ordinary recovery selections. Old/untagged
updates cannot enter recovery. Missing returns time out rather than becoming a
successful unlearning result. Mid-run resume of this protocol is explicitly
unsupported and rejected; do not load legacy checkpoints to pretend to resume it.

## Poisoning fix

`backdoor_enabled`, `backdoor_poison_rate`, `backdoor_target_label` and the optional
`backdoor_seed` now control a deterministic, client-local dataset view. The
training rate is the fraction of UNIQUE examples in the original client training
partition. Hash-based selection is stable across rounds. Retained clients are
never poisoned. Both training and target reverse optimization see the same poisoned
membership/labels; the base dataset and clean evaluation data are never modified.

The supported explicit trigger is `backdoor_trigger: invert`. Inversion is in raw
feature space, undoing and reapplying the known MNIST/CIFAR-10 normalization;
Purchase uses binary-feature inversion. This convention is a defined correction,
NOT a claim that every historical checkpoint used this exact operator. Unknown
datasets/triggers and invalid rates/labels fail visibly. ASR evaluation uses the
same operator on the full non-target test set, not test-subsampling labelled as a
training-poisoning sweep. Custom transforms require a separately validated operator.

## Associated fixes and explicit scientific choices

- Normal, forget and retained subsets preserve the actual configured sampler's
  original membership; they are not rebuilt from an unrelated IID partition.
- Virtual-worker phase and epoch settings reset on every new assignment.
- The corrected default is client-level removal (`mga_forget_scope: client`);
  `deleted_data_ratio` does not silently reduce the data of a fully removed client.
- Reference weights use frozen recorded participation-based scores, normalized
  over accepted retained clients for which records exist; unseen clients are not
  fabricated. Empty retained coverage fails. This is not model-version-delay weighting.
- Target budgets use the bounded `E_min + floor(log2(1+n_i))` rule unless an explicit
  fixed-budget ablation is configured. The count is frozen at request time.
- The tagged corrected reverse path starts from the common reference, accumulates
  the mean loss gradient across the target partition, normalizes it by `sqrt(E_i)`,
  and uses a scalar step. It rejects an infeasible proposal and returns the last
  feasible parameters. There is NO momentum or weight decay in this reverse path.
  BatchNorm buffers/dropout are held fixed in eval mode, with gradients enabled.
- `mga_lr` explicitly sets the scalar step. Otherwise it is base optimizer LR times
  `unlearning_lr_scale`. `mga_radius` defaults to 0.5 and should be explicitly set
  for a scientific experiment. These are not optimized to match the paper numbers.
- Returned target deltas are summed with positive `mga_alpha0/(1+score)` coefficients
  frozen at the request. They are not normalized; local feasibility does not imply
  an identical global radius bound. Integer model buffers retain reference values.
- The historical untagged trainer remains separate. Its momentum/projection/epoch
  behavior is not claimed equivalent to the corrected path.
- Failed training subprocesses raise before any old per-client weight file is read.
- `run_mga.py` seeds initialization, isolates paths, rejects pretrained initialization,
  records named checkpoint reads, and requires successful completion status.

## Evidence and limitations

`results/mga_events.jsonl` records the barrier, reference, every target dispatch and
return, completion, and recovery selections. Per-job poison membership hashes,
actual dataset reads, actual poison reads, accepted gradient steps and local losses
are in `results/mga_clients/*.execution.json`. `validated_result.json` additionally
reconstructs the target aggregation from returned weight files and evaluates the
initial/before/reference/immediate/recovery MNIST checkpoints.

These checks establish that the configured operations really ran. Good clean
accuracy or low ASR alone does not establish historical influence removal. Full
matched retraining and stop-participation controls, independent seeds, the other
datasets, and corrected timing/protocol claims still require new experiments.
