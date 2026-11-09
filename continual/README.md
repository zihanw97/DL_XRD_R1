# Continual-learning comparison

This directory runs the large-encoder, four-stage comparison with the
experimental training-label setting at 10.0% and logit-distillation temperature
`T=2`. The registered training-label mask is shared across every strategy,
seed, order and replay capacity.

The continual procedure includes a common dense geometry phase and feature
distillation.

From the package root:

```bash
python continual/launch_suite.py plan
python continual/verify_inputs.py
```

The first command prints the complete plan; the second verifies all 1,701
image fingerprints, registered label masks and every method/order/stage scope.
The plan includes six available controls, three seeds and
three block orders, the replay-capacity sweep, and the material-disjoint
secondary comparison: 72 four-stage chains in total. Five of the six controls
were displayed in the latest manuscript; the separate uniform experience
replay control remains available in the code.

To explicitly register and then run the comparison:

```bash
python continual/launch_suite.py prepare --run-dir continual/results/comparison --gpus 0
python continual/launch_suite.py schedule --run-dir continual/results/comparison
```

The GPU list can be comma-separated. The scheduler uses only listed GPUs that
are idle under its existing memory/utilization checks. One worker uses one GPU.

Data defaults are `data/cache/xrd_dataset_cache.npz` and
`data/partitions/continual_partition.csv`. `prepare --cache ... --assignments ...`
can relocate these files. The NPZ is a tensor-only image/label dataset, loaded
with `allow_pickle=False`. Its conversion
manifest and every registered image fingerprint are checked before training.
The SHA256-verified partition CSV defines the scan roles, block assignments,
and 10.0% experimental training-label mask. Hidden target-training labels are
`-1` in optimization and replay. `prepare` writes immutable run-profile CSVs;
workers retain their registered identities after block remapping. The
material-disjoint control uses the same training-label identities. Validation
labels are separate from the training-label budget.

Registered runs retain their source snapshots and training-label masks.

The procedure uses:

- The original six-convolution encoder, flattening to 381,024 values, a linear
  reduction to 2,048, SN-dense layers 512/256, and a 64-RFF SNGP head.
- Fresh Adam fitting at `5e-4`, warm Adam updates at `2e-5`, decay `0.997`,
  up to 800 epochs per base attempt, and validation-based stopping/selection.
- The existing bounded fresh-only retry at `1e-4` when its validation trigger
  fires, with all attempt costs retained.
- The unchanged dense AdamW phase at `5e-5`, weight decay `0.01`, up to
  60 epochs, compactness 10, supervised contrastive weight 0.1 and temperature
  0.1, and its validation-only acceptance gates.
- The main 128-image replay capacity, split 64 source / 64 target; source
  herding and target label-free diversity, preserving the registered 10.0% label mask.
- Proposed replay-logit distillation weight 3, temperature 2 with the `T**2`
  factor, and raw pre-head feature distillation weight 1. LwF uses logit
  distillation on current inputs without image replay, also at `T=2`.
  This logit-distillation setting is separate from the geometry contrastive
  temperature (0.1) and optional validation-fitted calibration controls.
- Epoch-wise GP-statistic resets and selected-state reconstruction from
  permitted current training plus prior replay, or accumulated training for
  joint retraining. Validation/test inputs never update precision.
- Saved predictions, exact sample IDs and raw 256-dimensional pre-head
  embeddings after each selected stage, with timing, retention, transfer and
  calibration reporting.

`recipe_provenance.json` supplies the complete active configuration and hashes
of the original training modules. Internal `frozen_support` modules are kept
private to this procedure so edits to the separate fixed-model objective do
not change continual learning.
