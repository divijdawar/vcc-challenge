# PerturbedVAE inputs and tinygrad model

The data layer prepares the diagram's measured perturbed observation `x` and
independently sampled control observation `x^(u0)`. The model layer implements the
PerturbedVAE reference core in tinygrad, including its invariant and responsive
latent variables, decoder, losses, and an Adam optimization step. No biological
training run has been started. NumPy/SciPy are confined to observation preparation
and the batch-conversion boundary; model computation and gradients use tinygrad.

The perturbed input uses **VCC 2025 H1 training responses**; the unperturbed input
uses **VCC 2026 controls**. A/B/C are sampled uniformly per group as explicitly
experimental cross-context transfer. No context is relabeled or asserted to be H1.
VCC 2025 controls are retained for training reference statistics and evaluation;
they are never selected as the model's `x^(u0)`. K562/RPE1 are outside this setup.
No datasets are downloaded by this implementation.

## Run

From this project directory, using Python 3.11 or later:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[model]'
.venv/bin/python -m vcc inspect-data
.venv/bin/python -m vcc prepare
.venv/bin/python -m vcc validate
.venv/bin/python -m vcc split
.venv/bin/python -m vcc train --dry-run
DEV=CUDA .venv/bin/python -m vcc train
```

Before running, obtain the official **2025 training** file separately and place it
at `inputs/vcc2025/adata_Training.h5ad`, or edit `config.sources[0]["path"]`.
The [Arc dataset inventory](https://github.com/ArcInstitute/arc-virtual-cell-atlas/tree/main/virtual-cell-challenge)
lists `2025/train/adata_Training.h5ad`. The adapter expects raw counts in `X`, gene
symbols in the var index, `obs.target_gene`, `obs.batch`, and `non-targeting` controls;
`inspect-data` checks the local schema and a bounded count sample. A modified or
normalized-only file needs an audited adapter; it is not silently accepted.
Existing 2026 controls, gene list, target list, and manifest remain in `../data`.
No 2025 validation/test release or K562/RPE1 file is required for this initial run.

`prepare` reads the release manifest, target list, and exact gene axis from
`../data`. It writes to `artifacts/observations/vcc2025_vcc2026_v1/` and refuses to
overwrite an existing output. Source files are opened read-only. Source data,
configuration, code, metadata, and cache chunks have SHA-256 provenance records.
`validate` checks cache integrity and unchanged original inputs. It is not the
official submission validator. `validate-submission` checks the generated H5AD
against the release locally; the official external `vcc prep --dry-run` remains
the final format check.

Edit the single `class config` in the `src/config.py` to configure the project.
Settings are attributes such as `config.hidden_dim`, `config.learning_rate`,
`config.alignment`, and `config.seed`. Data paths, model dimensions, loss weights,
training options, sampling, and the 2025 source adapter all live in this
class. Runtime defaults read these same settings.
Modules use `from .config import config` directly. Edit its attributes before
starting a process; there is no config-file loader or command-line config path.
Data preparation copies source metadata before resolving release inputs, so it
cannot mutate `config.sources` or duplicate sources across calls. Run snapshots
remain JSON.

`train --dry-run` is read-only and reports missing files, preparation, or splits.
`train` derives input/output dimensions from prepared gene axes and the condition
dimension from the frozen **training** target vocabulary; `model-info` shows the
standalone configured model preview. It fits reference statistics on training rows
and 2025 fitting controls only, calibrates/fixes MMD bandwidth at initialization,
and uses prefetched groups with the captured trainer. `planned_steps` overrides the
default 20 epochs, where an epoch is the capped training-cell count divided by the
response rows per optimizer step; sampling is balanced over training targets.

Each run gets a fresh timestamped directory under `artifacts/runs/vcc2025_to_vcc2026/`:
`run.json` stores config, versions, hashes, axes, vocabulary, reference statistics,
and transfer policy; `metrics.jsonl` records training/development metrics;
`status.json` records completion/failure; `diagnostics/` records numerical failures.
Prediction bundles are saved periodically, on development improvement, and at
completion (`final/`). Development uses fixed held-out cells of training-seen targets
with fixed noise and sampling, at full regularization strength. It is a validation
proxy, **not** an official VCC score. Final-test and target-held-out development
rows never enter training losses, fitted statistics, or checkpoint selection.
Sparse shard reads may decompress neighboring held-out rows; those rows are not
selected for fitting. Ctrl-C saves an inference
snapshot after completed steps when possible; optimizer/RNG resume is not implemented.
Set `config.prediction_bundle` to the chosen completed bundle for prediction.

## Representation

- `raw_*.npz`: bounded CSR count chunks on the **full source measured gene axis**,
  retaining count values and dtype, including zero-library cells.
- `cells.csv`: original cell identifier/label, canonical target (empty for a
  control), context, optional experimental batch, raw library total, and pool.
- `source.json`: exact source, encoder, and output gene order; measurement masks;
  adapter settings; hashes; counts; normalization details; and chunk index.
- `report.json` / `resolved_config.json`: preparation status and provenance.

Normalized expression is computed chunkwise during validation and on first shard
access: `log1p(10000 * count / full_source_library_total)`, in float32 after
float64 normalization arithmetic. A bounded in-memory LRU retains raw and normalized
sparse shards; the default limit is 256 MiB **per reader**, configurable with
`config.observation_cache_bytes` (0 disables retention). Gene projections are built
once per reader. `reader.cache_info()` reports hits, misses, and retained bytes;
`reader.clear_cache()` releases retained shards. Gene selection/alignment happens **after**
computing the full-axis total. Raw counts are never overwritten. No full normalized
dataset or new disk cache is stored. A measured zero has mask 1; an absent gene has
value 0 and mask 0. The encoder receives expression concatenated with its mask.
The output measurement mask is supplied for masked reconstruction and
population losses. The reader also exposes `raw_output_counts`, aligned to the
decoder gene axis while retaining `raw_counts` on the full measured source axis.
Masked control/response reconstruction and expected-count population training are
implemented in the residual objective.

No HVG selection, PCA, or response summaries are fitted here. The default encoder
axis is the full release axis; provide `encoder_genes` after freezing target splits
to use a smaller explicit axis. An unmeasured output gene is recorded per source;
gradient-training support cannot be inferred from challenge controls.

## Reading controls

```python
from vcc.observations import ObservationReader

reader = ObservationReader("artifacts/observations/vcc2025_vcc2026_v1/vcc2026_A")
controls = reader.sample_controls(context="A", pool="inference", n=64, seed=42)
x_u0 = controls.x                    # float32 [64, encoder_genes]
encoder_input = controls.encoder_input  # [64, 2 * encoder_genes]
counts = controls.raw_counts         # sparse, full measured source axis
```

Contexts in the CLI come from the manifest. `A` is only an API example.
Dense reads are restricted to explicit minibatches (256 rows by default).
Zero-library cells remain in the raw cache but are excluded from sampling.

## Adding perturbed observations

The supplied VCC 2026 files contain controls only. The active `config.sources`
adapter selects the public VCC 2025 training H5AD as the sole supervised source.
Its file is not downloaded by the code. Pin count location (`X`, `layers/counts`, or
`raw/X`), gene identifiers, target/control labels, context, and batch.
Dense and CSR H5AD matrices are supported. CSC inputs are rejected rather than
materialized in memory. Counts require an explicit raw-count declaration and
must be finite, nonnegative, and integer-valued. This numerical check alone
cannot establish raw-count provenance; source metadata must establish it.

Gene identifiers are matched exactly in a declared common namespace; there is
no automatic symbol guessing or Ensembl-version stripping. Supply verified
`gene_aliases` and `target_aliases` explicitly. Duplicate/colliding gene identifiers
are rejected and require an audited resolution policy before ingestion.

Supervised source controls, if present, are deterministically split 80/20 into `fit` and
`evaluation_reference` per source/context with seed 42. Reference counts round
down, with at least one reference cell for pools of size >=2. A singleton remains
in `fit`; empty reference coverage must be reported by later evaluation. Challenge
controls use `inference` and are never treated as supervised responses. `ntc_id`
is not assumed to be an experimental batch.

`python -m vcc split` freezes a source-fingerprinted manifest before fitting:
20% of eligible targets for final testing; 10% of the remaining targets for
unseen-target development; 10% of cells within training targets for seen-target
development. It caps fitting at 256 response cells per training target and excludes
targets with fewer than 64 responses by default. These are internal splits of the
2025 **training release**, not the official 2025 validation/test datasets.
Training, development, and test rows are disjoint. Re-running split refuses to
overwrite it; choose a new manifest path when intentionally changing the protocol.
The sampler receives explicit row allowlists from this manifest:

```python
group = source_reader.sample_group(
    target="TP53", context="H1",
    permitted_response_rows=frozen_split_training_rows,
    control_reader=vcc2026_control_reader,
    transfer_context="A",  # explicit experimental transfer; never means H1 == A
    n=64, seed=42,
)
x = group["perturbed"].x
x_u0 = group["unperturbed"].x
```

The populations are sampled independently. Unperturbed cells come exclusively
from the selected VCC 2026 context's `inference` pool. Both readers must use the
same encoder/output gene order and identifier namespace. The default runner samples
each 2026 context with equal probability. The [organizer's context description](https://vcc-cli-wiki.virtualcellchallenge.org/#what-a-context-is-and-how-you-tell-them-apart)
says those labels are anonymous cell lines; the code infers no biological mapping.
`control_context_policy="verified_mapping"` is available only with an explicit
`verified_context_map`. Cross-dataset batch matching is not asserted, and
`batch_matched=False` is reported. An allowlist selects permitted responses; it
does not itself prove that the caller supplied a leakage-free split.

## Tinygrad model core

Physical source files live in `src/`; packaging exposes them as `vcc`. Install
the optional model dependency when ready (no install/download was performed
while writing this implementation):

```sh
.venv/bin/python -m pip install -e '.[model]'
.venv/bin/python -m vcc model-info
```

`model-info` reads the definition without loading data, initializing weights, or
starting training. The tinygrad dependency is pinned to revision
`9a0ed69465dd6841aeb45597efc34a9391d9874b`. The model was exercised locally on CPU
using an already available installation of that revision. GPU throughput and
fixed-weight parity against the authors' implementation have not been established.

Implemented modules:

- `src/conditioning.py`: frozen target vocabulary, one-hot/multi-hot labels,
  control encoding, and explicit rejection of unknown targets.
- `src/model.py`: shared encoder, Gaussian invariant and responsive-noise heads,
  reparameterized samples, condition-dependent triangular W, affine B, float32
  forward substitution, decoder, and control-only prior generation.
- `src/losses.py`: reconstruction with optional output masks, noise-space and
  invariant KL, released normalized pairwise alignment, raw-logit sparsity, and a
  separate invariant MMD option with one-time bandwidth calibration.
- `src/batching.py`: converts prepared observations, raw counts, and frozen training
  references into model tensors.
- `src/training.py`: one Adam update over accumulated group losses, warmup,
  on-device global gradient clipping, packed latent/solve/nonfinite diagnostics,
  and bounded fixed-shape TinyJit capture with an eager fallback.
- `src/prefetch.py`: bounded, ordered CPU observation prefetch, error propagation,
  and cancellation/cleanup, used by `Trainer.iter_observations`.
- `src/splits.py`: deterministic target/cell holdouts and source/disjointness checks.
- `src/experiment.py`: read-only preflight, sampling, fitted references/bandwidth,
  training, development checks, metrics, and prediction checkpoints.

The reference model computes:

```text
x -> shared encoder -> q(z_i | x), q(e_v | x)
x_control -> same encoder/invariant heads -> q(z_i | x_control)
u -> W_net -> strictly lower-triangular tanh-bounded W(u)
(I - W(u)) z_v = B(z_i) + e_v
concat(z_v, z_i) -> decoder -> reconstructed expression
```

The default dimensions are 85 for `z_i` and 20 for `z_v`. These are learned
representations; creating untrained model weights does not make them biologically
meaningful. The `src/config.py` sets widths, depths, loss coefficients,
and warmup explicitly. Set input/output dimensions and condition dimension from
the frozen gene axes and **training** target vocabulary before fitting.

Example construction and model calls, once observation batches exist:

```python
from vcc.conditioning import PerturbationVocabulary
from vcc.model import ModelConfig, PerturbedVAE
from vcc.batching import ModelBatch
from vcc.losses import PerturbedVAELoss, calibrate_mmd_bandwidth
from vcc.training import Trainer, RegularizationSchedule
from vcc.population import fit_population_reference

vocabulary = PerturbationVocabulary(tuple(frozen_training_targets))
model = PerturbedVAE(ModelConfig(
    input_dim=len(encoder_genes), output_dim=len(output_genes),
    condition_dim=vocabulary.dimension,
    append_measurement_mask=True,  # explicit VCC extension to the reference encoder
), device="CPU")
# Source controls define fitting statistics; VCC controls remain the model input.
reference = fit_population_reference(
    source_reader, response_context, frozen_training_response_rows,
    split_manifest_hash=frozen_split_hash, device=model.device,
)
batch = ModelBatch.from_observation_group(
    group, vocabulary, model,
    supported_output_mask=frozen_training_output_support,
    population_reference=reference,
)
forward = batch.forward(model, reconstruct_controls=True)
z_i = forward.latents.invariant
z_v = forward.latents.responsive

# Calibrate once at initialization on an explicitly permitted control batch.
# The fitting_control_means tensor must come from that separate calibration batch.
h = calibrate_mmd_bandwidth(fitting_control_means)
objective = PerturbedVAELoss(mode="residual", alignment="mmd", mmd_bandwidth=h)
trainer = Trainer(model, objective, RegularizationSchedule(planned_steps=planned_steps))
metrics = trainer.step([batch])

# Prediction requires only controls, condition vectors, and fitted model weights.
prediction = batch.predict(model)
expected_raw_counts = prediction.expected.counts
```

The example assumes frozen manifests and explicit transfer or verified context selection;
it does not create them. Reference normalized rowwise alignment is implemented for
reproduction. `Trainer` requires explicit opt-in to apply it to unpaired cells.
For our independently sampled VCC 2026 controls, use calibrated MMD or disable
alignment. MMD regularization still requires scientifically compatible populations.

`LatentNoise` accepts externally supplied Gaussian draws in both reconstruction
and generation. The trainer generates missing draws outside TinyJit and passes
them, changing batches, count constants, reference profiles, and the warmup tensor
as explicit inputs. Supplied noise remains caller-controlled.
`model.generate` retains reference prior generation of continuous expression.
`model.predict` is the residual path: it shares the inferred invariant sample and
base Gaussian noise between target/control decoder evaluations, applies their
difference to the original source-control expression, and returns expected raw
counts through `src/counts.py`.

## Training execution and prefetch

`Trainer` uses TinyJit by default (`config.training_jit = True`). Gradient
computation and Adam are captured separately: **one packed device-to-host read per
steady-state step** checks all loss/count/latent/gradient status before Adam runs.
Metrics and group diagnostics share that transfer. Clipping stays on the device;
clipped gradients and Adam updates are scheduled together. Invalid steps do not
change weights, Adam moments, or the completed-step counter. `Trainer(..., jit=False)`
retains the same computation without capture for numerical comparisons.

Raw-count validation, count baselines/budgets, and population-mask validation happen
at the batch boundary. Prepared observation batches use NumPy validation before
upload; manually constructed `ModelBatch` objects validate once on first use.
Treat model-batch inputs and prepared masks/counts as immutable; create a new batch
when changing data. Direct prediction/export keeps validation enabled.

TinyJit warms up on the first call, captures on the second, and replays thereafter
for each input layout. Keep group counts and shapes stable for best reuse. The
LRU retains at most `config.jit_cache_size` gradient captures (default 4); optimizer
capture is shared. Compilation/warmup must be excluded from throughput timing.
Loss mode, alignment, bandwidth, and weights are frozen for a trainer; the
regularization warmup coefficient is the changing tensor input.
There is no mixed precision or claimed RTX 3090 speedup without a GPU measurement.

To overlap sparse reads and normalization with training, supply a **lazy** iterable
of CPU observation-group lists, one list per optimizer step:

```python
from contextlib import closing

def observation_steps():
    for requests in frozen_training_schedule:
        # Each request includes a frozen response allowlist, explicit transfer context,
        # VCC control_reader, target/context, n, and a changing sampling seed.
        yield [source_reader.sample_group(**request) for request in requests]

with closing(trainer.iter_observations(
    observation_steps(), vocabulary,
    supported_output_mask=frozen_training_output_support,
    population_references={response_context: reference},
)) as training:
    for metrics in training:
        # Persist/print metrics at the run's desired logging interval.
        pass
```

`config.prefetch_steps = 2` bounds completed CPU steps waiting in the queue; one
additional source step may be in progress. Set it to 0 for synchronous iteration.
GPU tensor conversion stays on the consumer/training thread. Worker errors are
raised by the consumer; closing cancels queued work and lets an in-flight read
finish. Iteration stops at the planned training-step limit. Split selection and
sampling policy remain caller responsibilities. The training CLI uses this API;
resumable optimizer checkpoints are not implemented.

## Residual training and count export

`config.training_objective = "residual"` gives response and control reconstruction
equal weight, averages both populations' KL terms, and adds population-effect
error on expected counts. The population coefficient remains fixed during warmup.
Invariant MMD requires a calibrated, frozen bandwidth; the default is not guessed.
Use `mode="reference"` explicitly for the original reconstruction objective.

`fit_population_reference` uses only source fitting controls and the caller's
frozen-training response allowlist. It excludes each target's own gene when
fitting the per-context effect scale. Keep evaluation-reference controls and
held-out responses out of this helper. The split hash records provenance but does
not replace the caller's responsibility to supply leakage-free training rows.
At loss time, unmeasured/unsupported genes and the group's own target are masked.

For explicitly cross-context groups, the predicted effect is the generated 2026
profile minus its sampled 2026 control profile. The measured effect is the 2025
response profile minus the training-fitted 2025 control reference. Comparing these
effects avoids training the target-context prediction to copy absolute H1 baseline
expression. This still assumes effects can transfer between contexts; MMD across
different cell lines is also an experimental regularizer, not evidence that their
invariant biology is identical. The reference population-loss behavior remains
available for groups using verified context correspondence.

The shared converter clips log-normalized residual expression to a legal range,
reverses the logarithm, normalizes supported genes into probabilities, and keeps
their original integer count budget. Unsupported genes retain their original raw
counts. Zero supported weights fall back to the original supported proportions.
Training uses continuous expected counts so gradients flow. Export samples integer
counts with NumPy's multinomial; exact null changes copy source rows without
resampling. All source controls and input files remain untouched.

After fitting, save a prediction bundle with the exact training vocabulary,
encoder/output gene axes, and support mask derived from gradient-training
measurements (including measured zeros). Never infer support from challenge
controls. Checkpoint parameters and count policy are validated when loading:

```python
from vcc.checkpoint import save_prediction_bundle

save_prediction_bundle(
    "artifacts/models/final", model, vocabulary, encoder_genes, output_genes,
    frozen_training_output_support, optimizer_steps=trainer.completed_steps,
    training_metadata={"split_manifest_hash": frozen_split_hash},
)
```

Set `config.prediction_bundle = "artifacts/models/final"`, configure the release
paths and matching prepared controls, then run:

```sh
python -m vcc predict
python -m vcc validate-submission
```

This writes one sparse H5AD plus a JSON report, using exactly 400 cells per target
and context under the current release contract. Context identities, targets, and
gene order come from the selected release. The upload H5AD contains predictions
only: **no non-targeting control rows**. Local checks include gene/label coverage,
unique row IDs, integer finite nonnegative values, per-cell totals, total cells,
and the stored-entry cap. The report provides the official `vcc prep --dry-run`
command; the project does not run it, package/upload, or submit automatically.
See `evaluation/vcc2026/README.md` for the external `.vcc` packaging step.

The current vocabulary uses identifiers and rejects unseen targets. During export,
a target without a trained condition or usable target features receives sampled
source-control counts as an explicitly reported no-change fallback. ESM/feature
conditioning is not implemented; fallback is not a learned unseen-gene prediction.

No optimizer/RNG resume checkpoint, mixed precision, ESM features,
context descriptors, or conditional noise prior is
claimed. Prediction bundles are inference checkpoints only. No real training run
or official evaluator/CLI validation has been performed. See
`THIRD_PARTY_NOTICES.md` for attribution and reference differences.
