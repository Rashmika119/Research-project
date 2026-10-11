# Historical research archive

Only the three-model 100/10 study is active. See [the current guide](../FULL_DATASET_300.md)
and [cleanup report](../CLEANUP.md). The nine files in `configs/` are historical
protocol records, copied byte-for-byte from their original `experiments/configs/`
paths. Do not use them to configure the current runner.

Historical documents remain at their original paths to preserve links and evidence:

- [Stage 1 protocol](../STAGE1_PROTOCOL.md) and [results](../STAGE1_RESULTS.md).
- [Stage 2 results](../STAGE2_RESULTS.md) and [warmup protocol](../WARMUP_ABLATION_5000.md).
- [Warmup instructions](../WARMUP_STAGE_INSTRUCTIONS.md) and [README](../WARMUP_STAGE_README.md).
- [Implementation history](../IMPLEMENTATION.md), [reliability findings](../RELIABILITY.md),
  and [earlier notes](../HISTORICAL_NOTES.md).

The two protected historical notebooks remain in `notebooks/` unchanged. Their
old commands and links describe their original source revisions; they are not
launch instructions for this cleanup branch. Their generators and old runners
can be recovered through Git. No historical notebook was relocated or regenerated.

Pre-cleanup source commit: `d9348c744da7c30c47e585de9e9ed4674a889399`.
For read-only recovery use `git show <commit>:<original/path>`. To run historical
code in isolation, create a separate worktree at the commit recorded in that
experiment's provenance, for example:

```bash
git worktree add ../Research-project-historical d9348c744da7c30c47e585de9e9ed4674a889399
```

That commit preserves the files removed during cleanup; older runs may require
an earlier commit. Preserve original package/runtime settings as well. Never
bypass source identity checks or point new runs at old checkpoint directories.
Google Drive experiment data is outside the cleanup scope and was not touched.
