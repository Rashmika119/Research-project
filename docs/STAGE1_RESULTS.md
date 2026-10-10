# Completed 1,000-entity stage: user-reported results

These values were supplied by the researcher in the next-stage task. They are
preserved as historical evidence, not recomputed or asserted statistically
conclusive here. The underlying Drive checkpoints/results remain untouched.

| Architecture | Mean validation MRR | Mean test MRR | Test Hits@10 |
| --- | ---: | ---: | ---: |
| Original | 0.06538 | 0.06355 | 14.73% |
| Residual | 0.08972 | 0.10318 | 16.28% |
| No-Refinement | 0.08066 | 0.09797 | 15.70% |
| Original No-Warmup | 0.06339 | 0.06016 | 13.95% |
| Residual No-Softprompt | 0.08902 | 0.10493 | 15.31% |

The researcher reported extensive score ties in both Original configurations and
lower runtime for Residual No-Softprompt. Those observations motivate the new
warmup ablation; three seeds do not establish statistical significance. Residual
had the highest mean validation MRR. No-Refinement is retained for a future study.

The previous stage used a 1,000-entity traversal subgraph and seeds 0/1/2. A local
data audit reproduced its 1,022 training facts, 55 validation and 86 test facts.
The new stage changes both entity scope and edge-retention strategy. Do not
interpret direct old/new MRR differences as a controlled model improvement.

Preserved protocol: [STAGE1_PROTOCOL.md](STAGE1_PROTOCOL.md) and
`experiments/configs/research_protocol_1000_historical.yaml`. The user's historical
`notebooks/research_pipeline.ipynb` is unchanged, including manual cells/outputs.
Its SHA-256 at this task's start and end is
`24fc8df77eff6473b016c7ee84cc7e1384af88d33924ad8bc5a917501d2270c8`.

Use the historical code revision with historical notebooks/checkpoints to reproduce
old runs. Active registrations and data preparation now implement the new stage.
