# RoBERTa loading and checkpoint resume investigation

Historical reliability work for the earlier five-variant stage is recorded below.
Its loader, complete RNG restoration and strict deterministic checks are retained
in the [current four-configuration study](WARMUP_ABLATION_5000.md). The new notebook
also isolates corruption fixtures from persistent Drive model backups.

This change preserves all five variants, scratch KG initialization, frozen
pretrained RoBERTa, equations, 5+25/0+30 budgets, fixed LR, BCE, ComplEx scoring,
original splits and validation-only selection. Short synthetic diagnostic runs
below are tests, not additions to the research experiment budget.

## Failure 1: invalid safetensors header

The supplied traceback fails while parsing weight bytes, before moving the model
to CUDA. A safetensors file begins with an eight-byte little-endian header length;
the reported error means that value exceeds the format's allowed header size.
LFS pointer text, HTML responses or corrupted initial bytes can cause it. A merely
truncated payload can cause a different deserialization error. CUDA is not an
explanation for a malformed file header. See the [format specification](https://github.com/huggingface/safetensors#format).

Confirmed repository defects: the old loader delegated cached files directly to
Transformers with no integrity check; the notebook placed HF_HOME on mounted
Drive; default revision resolution was mutable. **The original failing Colab
file is not available locally.** Its exact path/content and whether corruption
originated from download, Drive, a pointer file or another source are unconfirmed.
No package incompatibility has been established from the supplied traceback.

The chosen model is the same `roberta-base` checkpoint, revision
`e2da8e2f811d1448a5b465c236feacd80ffbac7b`. Its published model SHA256 is
`5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81`.
This matches the actual existing local file. The [pinned Hub file page](https://huggingface.co/FacebookAI/roberta-base/blob/e2da8e2f811d1448a5b465c236feacd80ffbac7b/model.safetensors)
provides the upstream checksum. Configuration/tokenizer files are pinned and
hashed too. Versions are printed and recorded; no arbitrary library downgrade
is used to conceal corrupt bytes.

`models/pretrained.py` validates size, initial bytes, header length, safetensors
layout/offsets and full SHA256. LFS pointers, HTML/XML, truncated files, invalid
JSON and payload checksum mismatches produce file-specific diagnostics. Only a
confirmed invalid Hub entry triggers `force_download=True` for that filename;
the replacement must also pass validation. No whole-cache deletion, alternate
model, random encoder initialization or `.bin` fallback exists. Hub's supported
[download API](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download)
accepts explicit revision, cache directory and per-file force download.

In Colab, HF_HOME is `/content/hf_cache`; both Hub cache environment names point
to `/content/hf_cache/hub`; stale TRANSFORMERS_CACHE is removed before imports.
Downloads are staged atomically into `verified_snapshots/roberta-base/<revision>`.
RESEARCH_MODEL_BACKUP points to `verified_roberta_backup` on Drive: verified files
are copied there and can repopulate local storage after a reset. Restored files
are revalidated. Transformers opens only the local staged copy, never the Drive
backup. The old Drive HF cache is preserved for diagnosis, not reused silently.
Repeated valid loads do not redownload. Integrity checks do read file contents;
there is a disk-I/O cost, especially when checking the persistent backup.

An explicit local model directory is checked but never repaired or modified.
Remote model names/revisions outside the pinned protocol are rejected explicitly;
custom local directories must provide the supported RoBERTa safetensors/tokenizer
files. The stock MLM checkpoint does not contain AutoModel's unused pooler;
missing pooler weights are allowed because every research path uses
`last_hidden_state`. Missing encoder weights or shape mismatches are fatal.

For the original failing file, run the read-only diagnostic in Colab:

```bash
python run_pretrained_check.py --inspect-file /path/to/original/model.safetensors --report original_file_diagnosis.json
```

It reports resolved path, length, first bytes, header length and checksum. For a
full load and forward gate:

```bash
python run_pretrained_check.py --device cuda --report pretrained_report.json
python run_scratch_smoke_test.py --real-lm-only --device cuda
```

The latter runs all five architectures through actual pretrained forward/backward
computation, checks component gradients, absence of a soft prompt in variant 5,
and unchanged frozen LM weights after an optimizer update.

## Failure 2: CUDA uninterrupted/resumed mismatch

The reported 2.98e-8 maximum difference is consistent with floating-point reduction
ordering but is not proof. CPU success alone cannot identify its GPU cause.
Inspection found that the old implementation already restored trainable model
parameters/buffers, Adam state, Torch CPU/all-CUDA RNG, independent Python negative
sampling RNG and CPU shuffle-generator state. It restored history, epoch, update
count, best selection, and LM identity. **Global Python and NumPy RNG were missing.**
Those global streams are not consumed by the current training loss/negative
sampler, so their omission is a reproducibility gap, not demonstrated to explain
the observed three tensor elements. Initialization randomness is deliberately
separated from the training stream before checkpoint restoration.

The old deterministic setting covered cuDNN only. This RGAT's installed PyG sum
aggregation uses `Tensor.scatter_add_`; attention uses grouped softmax reductions.
cuDNN determinism does not govern these. PyTorch documents a deterministic CUDA
scatter_add implementation under [torch.use_deterministic_algorithms](https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html).
The new training default enables that setting with `warn_only=False`, disables
cuDNN benchmarking, and configures CUBLAS_WORKSPACE_CONFIG before computation.
The model's attention implementation and math are not replaced. Third-party
kernels and different CUDA/library versions still require real GPU validation.

New `training/reproducibility.py` saves/restores and immediately audits:

| State | Treatment |
| --- | --- |
| Model parameters and buffers | Every non-LM state_dict entry, exact after loading |
| Frozen RoBERTa | Reloaded from verified revision; identity checked, not duplicated in checkpoints |
| Optimizer | Adam moments, steps and parameter groups, exact after loading |
| Scheduler / gradient scaler | Explicitly absent; fixed LR and full precision |
| Python / NumPy global RNG | Full state, including NumPy cached Gaussian |
| Torch RNG | CPU and all visible CUDA generators; device count must match |
| Negative sampling / shuffling | Independent random.Random and CPU torch.Generator |
| Module modes | Training/evaluation flags restored and audited |
| Training progress | Committed epoch, history, step count, initial validation, best weights/selection |
| Numerical execution | Determinism, cuDNN, TF32, matmul, SDP and cuBLAS policy recorded and matched |

Resumable `last.pt` is an atomic epoch-end commit; a partial epoch is replayed
from the preceding commit. There is no gradient accumulation or scheduler cursor
to restore. Next iteration clears gradients before backward. `best.pt` remains
selection-only; it is not a resume checkpoint. Saving snapshots performs no
optimizer step. Exact state comparisons occur before any resumed computation;
`resume_audit.json` records success. Restoring selected-best artifacts is I/O only.

**No tolerance was loosened.** Float tensors, integer tensors, optimizer steps,
RNG state and discrete progress remain exact comparisons. A test deliberately
changes weights and discrete values and checks rejection. The dedicated runner
compares two uninterrupted controls and an interrupted/resumed run, with per-tensor
maximum absolute/relative deltas, losses, optimizer state and evaluation metrics.
Three repeats alternate interruption after warmup and after an integrated epoch;
mini-batches exercise multiple updates and shuffle/negative sampling state.
The single-repeat real-LM check interrupts after epoch 2, so it restores optimizer
moments for the integrated projections/refiner as well as structural parameters.
Strict CUDA kernel failures remain visible rather than being waved through.

```bash
python run_scratch_smoke_test.py --offline-only --device cpu
python -m unittest tests.test_research_pipeline.PipelineTests.test_resume_matches_uninterrupted -v
python run_resume_diagnostic.py --device cuda --repeats 3 --output-dir experiments/research/cuda_resume
python run_resume_diagnostic.py --device cuda --real-lm --variant residual --repeats 1 --output-dir experiments/research/real_cuda_resume
```

Use fresh diagnostic directories for reruns. `--device cuda` rejects unavailable
CUDA; it never reports CPU work as GPU verification. `--device cpu` on the resume
runner hides CUDA before importing Torch. The smoke runner's offline default also
uses a CPU child process; real-LM mode runs independently (`--real-lm` remains an
alias). For the direct unittest CPU command on Colab prefix `CUDA_VISIBLE_DEVICES=""`.

The new RNG schema and source identity require a new research RUN_TAG. Earlier
checkpoints are preserved; they cannot safely acquire missing RNG state retroactively.

## Notebook and changed files

Both `notebooks/build_colab_notebook.py` and its generated notebook are updated.
Section 6 contains independently runnable preflight, CPU regressions, GPU
real-RoBERTa, repeated targeted CUDA resume/control, and real-RoBERTa GPU resume
cells. Each experiment launch also enforces preflight, so disabling optional
test flags or skipping their cells cannot bypass model loading validation.
Default research budgets and all fifteen configurations remain unchanged.

`training/process.py` streams stdout/stderr and saves full logs. On failure it
reports command, exit code, log path and the final output, rather than only a
CalledProcessError. Logs and diagnostic reports live under the Drive run directory.

| Failure/area | Changed files |
| --- | --- |
| RoBERTa | models/pretrained.py (new), models/frozen_lm.py, run_pretrained_check.py (new), requirements.txt |
| Resume | training/reproducibility.py (new), training/research_pipeline.py, training/experiment_io.py, run_resume_diagnostic.py (new) |
| Gates/notebook | run_scratch_smoke_test.py, training/process.py (new), notebooks/build_colab_notebook.py, notebooks/research_pipeline.ipynb |
| Tests | tests/test_pretrained.py (new), tests/test_reproducibility.py (new), tests/test_process.py (new), tests/test_research_pipeline.py, tests/test_notebook.py |
| Documentation | README.md, CLAUDE.md, docs/IMPLEMENTATION.md, docs/RELIABILITY.md (this file) |

## Verification record

Local environment: Windows, Python 3.14.3, PyTorch 2.11.0+cpu,
Transformers 5.13.0, huggingface-hub 1.22.0, safetensors 0.8.0, PyG 2.8.0.post1.
This differs from the supplied Colab Python 3.13.15/cu130/T4 runtime.

- Targeted CPU resume regression: passed, exact weights and extended state checks.
- Full offline CPU suite: 17 tests passed, including corruption/recovery,
  unrelated-file preservation, RNG replay, log failures, notebook schema/ordered
  execution and mandatory preflight blocking (external calls mocked).
- Three repeated CPU control/resume experiments: passed exactly for losses,
  weights, optimizer/RNG/progress and metrics; maximum tensor difference zero.
- Fresh Hub download into a separate project cache: all six file hashes passed;
  actual pretrained CPU loading and forward passed. The 498,818,054-byte weight
  file has a 24,202-byte header and 203 serialized tensors. Initial sandbox
  network access was blocked; an approved download succeeded using the Hub HTTP
  transport (`HF_HUB_DISABLE_XET=1`) after stopping an incomplete Xet attempt.
  This transport choice did not change model, revision or weights and is not
  asserted to be necessary in Colab. The notebook keeps the default Hub transport.
- Actual pretrained CPU smoke: all five variants passed forward/backward,
  component-gradient and frozen-weight checks through the new verified loader.
- Actual pretrained residual CPU control/resume at both the warmup and integrated
  epoch boundaries: passed exactly, including losses, weights, optimizer states,
  RNG and evaluation metrics; maximum tensor difference zero.
- Python syntax: 67 files parsed; existing ComplEx and tie-ranking checks passed.
- Explicit CUDA smoke and resume commands: rejected CUDA as unavailable. These
  are **not GPU passes**; repeated CUDA tests and GPU training remain unverified.

Local logs/reports are under `experiments/research/reliability/` (ignored by Git):
`offline_tests.log`, `pretrained_cpu.json`, `pretrained_cached_cpu.json`,
`real_roberta_cpu.log`, `resume_cpu/resume_diagnostic.json`, and
`real_resume_cpu/resume_diagnostic.json`, and
`real_resume_cpu_integrated/resume_diagnostic.json`. Synthetic metrics in those reports are
mechanics checks, not research results.

No Colab execution, full-data experiment or performance improvement is claimed.
Full acceptance of the two original Colab failures requires running the GPU cells
successfully in that environment and inspecting the original bad file. No CUDA
tolerance has been justified or introduced by CPU-only evidence.
