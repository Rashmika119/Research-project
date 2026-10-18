# Two-model cleanup

The active code contains only KG-only RGAT/ComplEx baseline and Shared C gated.
Shared C equations,parameter names/order,scratch initialization and numerical
settings are retained. Registry and constructors reject retired choices.

Removed independent-refiner code,A/B and ungated paths,older study launchers/report
modules,redundant notebook generators and retired-model tests. One runner,notebook,
YAML protocol and two-model reporting path replace the previous study entry points.

Historical notebooks (including embedded outputs),configs and documents were moved
out of active paths and preserved byte-for-byte under docs/archive/pre_two_model.
Its manifest records SHA256 values and pre-cleanup source commit. Recover earlier
source from Git; archived notebooks require original source/environment.

No data,pretrained caches,result directories or checkpoints were deleted. New runs
use a new protocol/root/tag; old checkpoints are not migrated.

See [removed paths](cleanup_manifest.json) and
[preserved hashes](archive/pre_two_model/manifest.json).
Active notebook: notebooks/baseline_vs_shared_c_full.ipynb.

## Verification on 2026-10-18

`python run_static_checks.py`: 22 checks discovered,21 passed,1 skipped.
The full local dataset audit was skipped because sandbox access to train.txt
was denied and additional access was declined. The research runner still requires
full-data verification before training. No corrupt-data failure was bypassed.

Passed checks include exact model/fusion contracts,two-path shared gradients,
checkpoint/Adam/RNG restoration,pretrained corruption handling,ComplEx algebra,
filtered tie ranking,report generation/rejection tests,mocked two-run orchestration,
notebook execution and dependency-lock format,plus archived evidence hashes.
All54 active Python files parsed,CLI help lists only baseline/shared-c-gated,
and git diff whitespace checks passed.

No research training or optimizer steps were executed. Actual pretrained GPU
smoke tests,CUDA resume comparisons and full-dataset memory capacity remain
unverified in this environment. Use the independent diagnostics in RELIABILITY.md
and the new Colab notebook for those measurements.
