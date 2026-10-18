# Reliability and verification

`python run_static_checks.py` runs offline CPU checks with no optimizer steps or
pretrained downloads: shared-layer gradients, equations, pooling cache, frozen LM,
checkpoint/RNG restoration, strict identities, report validation, dataset counts
when local data exists, notebook syntax and mocked execution. Tiny LM and report
fixtures test mechanics only. Tests write fabricated records to temporary folders.

Pinned RoBERTa revision:`e2da8e2f811d1448a5b465c236feacd80ffbac7b`.
Pretrained loading checks safetensors headers and weight completeness, rejects
LFS pointers/truncated files, supports verified backup recovery and targeted
replacement of corruption. It never substitutes random weights. Only frozen
pooling is cached; projection and graph embeddings remain live.

Atomic epoch checkpoints save model/buffers,Adam moments,module modes,Python/NumPy/
PyTorch CPU/CUDA RNG,negative sampler/shuffle RNG,progress,history and best state.
Scheduler/scaler are absent. Resume verifies source/data/text/settings/runtime/
numerical identities, restores exactly and replays an uncommitted epoch. Archived
checkpoints require original source; use a fresh tag for the cleaned protocol.

These independent GPU diagnostics are opt-in, not silently run by the notebook.
Smoke/resume commands take optimizer steps on tiny synthetic facts, not benchmarks:

```bash
python run_pretrained_check.py --device cuda
python run_scratch_smoke_test.py --real-lm --device cuda
python run_resume_diagnostic.py --variant shared-c-gated --device cuda --real-lm --repeats 3 --output-dir experiments/research/diagnostic_shared_fresh
python run_resume_diagnostic.py --variant baseline --device cuda --repeats 3 --output-dir experiments/research/diagnostic_baseline_fresh
```

CPU equivalents use `--device cpu`. Resume diagnostics compare two uninterrupted
controls and interrupted/resumed training, including weights,moments,RNG,
non-timing histories and metrics, exactly. Diagnostic repeats use different seeds;
the full research runner permits only0. Differences write JSON; no silent tolerance
relaxation. Fresh diagnostic directories prevent overwriting earlier runs.

Child errors stream tracebacks and report command,exit code and log path. CUDA OOM
writes progress/failure diagnostics without changing settings. Strict determinism
may expose unsupported CUDA operations. CPU checks do not certify CUDA behavior,
real pretrained GPU computations or15GB T4 capacity; verify these in Colab.
