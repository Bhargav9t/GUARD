# GUARD Model Checkpoints Directory

This directory stores saved PyTorch weights (`.pt` / `.pth`) during training runs.

## Checkpoint Files:
- `guard_best.pth`: Model weights achieving the highest validation AUC-ROC / lowest EER.
- `guard_latest.pth`: Most recent epoch weights for resuming interrupted training.
- `guard_epoch_N.pth`: Periodic snapshots taken at designated intervals.
