# Completed 5,000-entity single-seed pilot

Provenance: the researcher's saved outputs in the manually edited
`notebooks/warmup_ablation_5000.ipynb`, inspected during full-study planning.
These results were not rerun during implementation. Colab output records code
commit `9a0420d58ce4b815bf0a9805e2fb224da512588f`, Python 3.13.15,
PyTorch 2.11.0+cu130 and Tesla T4. Run tag: `warmup5000_induced_v7_run1`.

Seed 0 only; 5,000 entities, 72,374/4,717/5,465 train/valid/test triples;
30 full-batch updates; five warmup epochs where enabled; text batch 256,
evaluation batch 512 and length 64. Saved titles referring to three seeds/twelve
runs are stale: executed configuration/output shows one seed and four runs.

| Configuration | Validation MRR | Test MRR | Test Hits@10 | Best epoch | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Residual warmup | .050142 | .049499 | 10.15% | 30 | 2269.06 |
| Residual no warmup | .035690 | .035612 | 8.43% | 25 | 2663.23 |
| No-Softprompt warmup | .069402 | .071099 | 12.90% | 25 | 186.37 |
| No-Softprompt no warmup | .061043 | .061820 | 11.54% | 29 | 170.06 |

No-Softprompt warmup has the highest validation MRR in this pilot. Its final-epoch
validation MRR was .060159, below its selected epoch-25 result. No-Softprompt
without warmup ended at .060871. These observations motivate examining learning
curves and seed variability, not a claim that 300 epochs will improve results.
No-Softprompt peak allocated GPU memory was about 4.0 GB on this subset; this does
not verify full-data capacity. Total time includes cache/setup/evaluation overhead.

Only one seed was run; differences are not statistically conclusive. Full-data
ranking has a different candidate scope and graph, so pilot scores are not
directly comparable with the new study. Historical checkpoints/outputs stay intact.

Protected notebook SHA256:
`d5137bc4e16917fb8b46bf641cdbd532815c5bcfa5cc32f1fbfb031e979973f4`.
Original research notebook SHA256:
`24fc8df77eff6473b016c7ee84cc7e1384af88d33924ad8bc5a917501d2270c8`.
