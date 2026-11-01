# PerturbedVAE-inspired VCC 2026 model in tinygrad within $50

## 1. Objective and implementation boundaries

Build a **challenge-adapted, PerturbedVAE-inspired model** in `/Users/divij/Desktop/vcc/PerturbedVAE`, covering data preparation, training, local evaluation, and submission-file generation. Read existing inputs from `/Users/divij/Desktop/vcc/data`; preserve the existing data, analysis scripts, and reports. Put new code, configurations, caches, and run artifacts under `PerturbedVAE/`.

Implement the model, differentiation, optimization, and prediction in **tinygrad**. NumPy/SciPy/AnnData may handle data preparation and export. Separate optional environments may use PyTorch for frozen ESM feature extraction and reference parity checks, and the official evaluator may use its own dependencies. These are auxiliary processes; the trainable model and its inference path must not depend on PyTorch.

The paper explicitly excludes unseen-gene and cross-cell-type generalization from its claims. Preserve its main modeling ideas while labeling challenge additions separately. The diagram's identifiable responsive block and block-identifiable invariant block depend on theoretical assumptions; context conditioning, changed alignment, and residual decoding do not automatically inherit those guarantees. Neither biological causal discovery nor challenge performance is guaranteed. [Paper, theory and limitations](https://arxiv.org/html/2605.19343v1)

Use the authors' MIT-licensed implementation at commit `12cf29f4410464c15c142c245b171eed2cca4971` as the reference. Retain attribution and implement independent splits and evaluation; do not rely on the upstream placeholder evaluation entry point. [Authors' repository](https://github.com/week3ndzZ/PerturbedVAE)

The current Python command interface runs from the project directory or after installing its package (standalone official-metric evaluation remains separate):

```text
python -m vcc inspect-data
python -m vcc prepare
python -m vcc validate
python -m vcc split
python -m vcc train --dry-run
python -m vcc train
python -m vcc predict
python -m vcc validate-submission
```

Record resolved configurations, data hashes, gene order, feature and split manifests, software revisions, seeds, checkpoints, elapsed GPU time, and billable instance time. Keep reference and challenge modes explicitly separate.

## 2. Active data preparation and evaluation protocol

The initial runnable setup uses exactly two releases:

| Dataset | Role |
|---|---|
| VCC 2025 H1 **training release** | Sole perturbed-cell source; its controls provide fitting-reference statistics only |
| VCC 2026 supplied controls | All unperturbed model inputs, calibration controls, and prediction inputs |

The 2025 training file is supplied separately at `inputs/vcc2025/adata_Training.h5ad` (or the path in `config.sources`). It is not downloaded automatically. Existing 2026 controls and release metadata remain in `../data`. K562 and RPE1 are not dependencies of this setup. The official 2025 validation/test expression files are not used.

**Context policy:** Sample the supplied 2026 contexts uniformly per group as explicitly experimental cross-context transfer. A/B/C are anonymous different cell lines; never infer or assert that H1 corresponds to one of them. Preserve each control cell's context label. The alternative verified-mapping policy requires a separately verified mapping. MMD across these contexts is an experimental invariance regularizer, not evidence of biological equivalence.

**Preparation:** Audit the configured raw count location (`X` for the default 2025 adapter), gene symbols, `target_gene`, `non-targeting`, and `batch` with `inspect-data`. Normalize by each cell's full source library before gene projection. Decode onto the full 2026 release gene axis; zero-fill missing encoder genes with measurement masks and exclude missing outputs from losses. The initial runner uses the full prepared encoder axis and fits no HVGs/PCA. Unsupported output genes preserve source-control counts during export.

**Freeze splits using seed 42 before fitting:**

1. Use only eligible nonzero 2025 training-release responses (at least 64 cells per target by default). Reserve 20% of targets for final unseen-target testing.
2. Reserve 10% of the remaining targets for unseen-target development. Neither target holdout enters gradient updates, reference fitting, or fitted response summaries.
3. Hold out 10% of cells within each training target for seen-target development; fit on at most 256 other cells per target. Split metadata fingerprints the prepared source and rejects overlapping rows or altered membership.
4. Split 2025 controls 80/20 into fitting and evaluation-reference pools. The model never uses these as its unperturbed input. Fit the population reference and effect scale on 2025 fitting controls and frozen training responses only.
5. For cross-context population supervision, compare generated 2026 minus sampled 2026 control profiles against measured 2025 response minus fitted 2025 control profiles. Never force absolute H1 baseline expression onto 2026 controls.
6. Calibrate/freeze MMD bandwidth at model initialization using explicitly permitted 2026 controls. Keep generation noise and warmup tensors changing during TinyJit replay.
7. Select checkpoints with fixed seen-target development cells and fixed noise at full regularization strength. This is a development proxy, not an official VCC score. The training command does not evaluate target-held-out responses; final/unseen-target evaluation remains separate.

The initial identifier vocabulary contains training targets only. Unknown release targets receive the existing reported no-change fallback at export. ESM-conditioned unseen-target prediction, broad unseen-context validation, final all-target refitting, and optimizer/RNG resume remain future extensions; a successful training run does not establish those capabilities. Later multi-source experiment ideas in this document are deferred and do not change the active dataset selection above.

## 3. Model and training specification

### Reference core and parity

Port the pinned released core into tinygrad before introducing challenge additions:

- Hidden width 256, responsive dimension 20, invariant dimension 85; shared encoder backbone with separate Gaussian heads for structural noise and invariant latents.
- Standard-normal structural-noise and invariant priors; a perturbation-conditioned strictly lower-triangular matrix with `tanh`-bounded entries; a condition-independent affine map from invariant to responsive latents.
- Released decoder, MSE reconstruction, noise-space KL, invariant KL, normalized posterior-mean pairwise alignment, and sparsity reduction. Record activation, depth, and coefficient overrides explicitly rather than relying on constructor defaults.
- Compare fixed-weight outputs, individual loss terms, and gradients against the pinned implementation using identical externally supplied Gaussian draws. Use synthetic inputs and a small K562 subset. Document numerical safeguards separately; a full Norman2019 reproduction is outside this budget.

The reference pairwise alignment is retained for parity, not treated as proof that independently sampled controls are matched counterfactuals. [Released model](https://github.com/week3ndzZ/PerturbedVAE/blob/12cf29f4410464c15c142c245b171eed2cca4971/Norman_double/models/perturbedvae.py)

### Challenge features and conditional mechanism

- **Target features:** cached, frozen mean-pooled residue embeddings from the 8M-parameter ESM-2 checkpoint, excluding special/padding tokens, plus the target's baseline log-normalized expression, detection rate, and missing-feature indicators from the context's permitted controls. Record sequence identifiers and checkpoint revision. For sequences longer than the supported residue window, embed nonoverlapping windows and residue-weight their means; record this policy. Do not fine-tune ESM. Use training-only feature scaling. [ESM-2 checkpoint](https://huggingface.co/facebook/esm2_t6_8M_UR50D)
- **Context descriptor:** fit centered 64-component PCA on log-normalized H1/K562 fitting controls, restricted to shared measured encoder genes and balanced by context. A context descriptor is the mean of its control-cell PCA scores. Freeze PCA/scaling before held-out evaluation. Use the available rank and zero-pad if fewer than 64 components are possible. Measurement masks still accompany cell inputs.
- **Control condition:** append an explicit control indicator to conditioning features. For control rows, zero target-specific features and use the context descriptor. Include controls as an actual trained condition.
- **Posterior and priors:** encode diagonal Gaussians `q(z_i | x, mask, c)` and `q(e_v | x, mask, u, c)`. Use `p(z_i)=N(0,I)` and a learned diagonal structural-noise prior `p(e_v | u,c)=N(mu_p, diag(sigma_p²))`. Here `u` denotes target features and the control indicator; `c` is the context descriptor.

For both posterior reconstruction and prior generation, use the same structural map:

\[
z_v=A(u,c)^{-1}[B(u,c)z_i+e_v],\qquad A=I-W(u,c).
\]

During reconstruction, draw `e_v` from its posterior; during prediction and population training, draw `e_v = mu_p + sigma_p * epsilon` from its prior. Generate `W` and `B` from target/context features. Initialize the `W` output layer at zero and bound its strictly lower-triangular entries with `tanh`; initialize the conditional prior near unit Gaussian.

Compute the responsive KL **in structural-noise space**, conditional on the same sampled `z_i`. Because posterior and prior share the invertible affine map, their transformation Jacobians cancel:

\[
L_{\mathrm{KL},v}=\frac12\sum_j\left[
\log\frac{\sigma_{p,j}^2}{\sigma_{q,j}^2}
+\frac{\sigma_{q,j}^2+(\mu_{q,j}-\mu_{p,j})^2}{\sigma_{p,j}^2}-1\right].
\]

Do not apply a standard-normal KL to a learned nonstandard noise prior, or treat the transformed responsive covariance as diagonal. Average latent-dimension sums over cells. For challenge mode, clamp all posterior/prior log variances consistently to `[-12, 8]` in sampling and KL. Keep these operations and structural solves in float32.

Implement the 20-dimensional solve initially with differentiable forward substitution in tinygrad, without an explicit inverse. Triangularity guarantees invertibility but not numerical stability: log weight magnitudes, latent magnitudes, solve residuals, and nonfinite events. Stop and save diagnostics on nonfinite loss/gradients instead of silently continuing.

### Losses and training

\[
L=L_{\mathrm{reconstruction}}
+0.01L_{\mathrm{KL},v}
+0.01L_{\mathrm{KL},i}
+0.05L_{\mathrm{alignment}}
+10^{-6}L_{\mathrm{sparsity}}
+L_{\mathrm{population}}.
\]

- **Reconstruction:** average masked gene MSE per cell, then average cells; give perturbed and control reconstruction equal weight. Train controls through their posterior and control condition so the subtracted decoder branch is anchored. Average KL terms equally across the two populations. Sparsity is mean absolute value of effective lower-triangular `W` entries.
- **Alignment:** challenge mode uses biased, nonnegative empirical MMD² between invariant posterior means from each perturbed group and independently sampled controls in the same context/batch when available. Use a mean of three RBF kernels with bandwidths `h/2, h, 2h`; choose `h` as the median positive pairwise distance on a fixed fitting-control calibration batch at initialization, floor at `1e-3` (use `h=1` if no positive distances exist), and freeze it. Define each kernel as `exp(-squared_distance / (2 * bandwidth²))`. Do not normalize each latent mean to unit length. Report this as a distributional regularizer, not matched-cell supervision; population composition changes can still violate its assumptions. Monitor latent per-dimension variance and reconstruction, and compare against alignment disabled.
- **Population effect:** use the exact residual prediction and expected-count conversion below, before integer sampling. Pool expected counts over generated cells, normalize to 50,000, and apply `log1p`; subtract the fitting-control pseudobulk profile during training and the separate evaluation-reference control profile during validation/evaluation. Never use evaluation-reference controls to fit losses or cached training effects. Compare with the training response group's identically computed effect. Mask unmeasured output genes and the group's own target gene, and reduce equally over valid genes and target/context groups. For each source context, compute `s_c² = max(mean(training_effect²), 1e-4)` using training groups only and divide its squared effect error by `s_c²`. Use the same frozen scales for validation. Keep this loss distinct from the official six-metric score.
- **Sampling:** an optimizer step contains four uniformly sampled target/context groups with 64 cells each, plus 64 independent controls per group. Sample contexts uniformly, then targets within context. Sample with replacement when needed. Compute each group's MMD and population loss before accumulation; never mix unrelated groups into one alignment population. Process groups sequentially if required to stay below the memory cap.

Use Adam at `1e-4` and global gradient-norm clipping at `0.5` after accumulation. First establish float32 correctness. Then enable float16 neural-layer compute with float32 master parameters, Adam state, reductions, structural solves, and KL; use dynamic loss scaling, unscale before clipping, and skip/reduce the scale on overflow. Retain float32 if mixed precision fails the numerical or throughput checks.

Use a linear regularization warmup over the first 25% of the planned optimizer steps for KL, alignment, and sparsity; keep reconstruction and population coefficients fixed. Define an epoch as `ceil(N_training_perturbed_cells / 256)` optimizer steps under the balanced sampler. Train for at most 100 epochs; evaluate every five epochs and stop after four checks without lower source-group validation population-effect error following warmup. Persist the schedule and counters across resumes. Select checkpoints on this validation error, never training reconstruction alone.

### Prediction and count export

Inference accepts only source control cells, target identifiers, frozen features, a checkpoint, and a seed. For control count row `r` with total `L`, infer `z_i` from its log-normalized values and use shared `z_i` and standard-normal noise draws for target and control prior evaluations:

\[
\ell=\operatorname{clip}\left[
\log(1+10^4 r/L)+D(z_i,z_{v,\mathrm{target}},c)-D(z_i,z_{v,\mathrm{control}},c),
0,\log(1+10^4)\right].
\]

Convert this to expected counts using the checkpoint's supported-output set `S` and unsupported set `U`:

1. Copy `r_U` exactly. Set the remaining integer count budget `R = L - sum(r_U) = sum(r_S)`.
2. Compute nonnegative weights `w_S = expm1(ell_S)`, then probabilities `p_S = w_S / sum(w_S)`. If their sum is zero, use the original supported control proportions. If `R=0`, return zeros on `S`.
3. During population training, use expected supported counts `R * p_S`. During ordinary prediction export, draw `Multinomial(R, p_S)`. The original source-control library total is preserved, so sampling source rows supplies the library-size distribution without a second independent library draw.
4. An explicitly requested control condition, an exact all-zero supported decoder difference, or the no-change fallback copies the entire source row without resampling. This makes the null operation exact after export. The stochastic non-null path is a modeling assumption; compare expected effects and count-distribution diagnostics on development data.

Use the same expected-count conversion in population training and prediction, including unsupported-gene handling. Keep the null-copy branch as an export policy rather than a training shortcut. Reject nonfinite probabilities; do not silently repair overflow. Exclude zero-library source rows from generated-cell sampling, and report this exclusion.

Keep all supplied control rows unchanged as inference inputs. The official upload contains predictions only, with exactly 400 cells per target and context in one combined sparse file; do not append non-targeting control rows. Generate predictions by sampling source control rows reproducibly. For targets lacking usable target features and any observed training responses, copy sampled controls as a reported no-change fallback, labelled as the requested target. Missing ESM features alone do not require fallback when other usable features or training responses exist.

Validate exact gene order, labels, unique row identifiers, finite nonnegative integral values, and totals no greater than `10^6`. Reject an incompatible source-control input rather than modifying its supposedly unchanged rows. Enforce exactly 400 predictions per target/context, at most 400,000 total cells, and at most 4,750,000,000 stored entries. Validate/package the sparse H5AD with the official CLI; no automatic upload is authorized. The stricter upload contract takes precedence over the generic scorer's matrix contract. [Official submission requirements](https://vcc-cli-wiki.virtualcellchallenge.org/#submission-requirements-2026), [metric definitions](https://github.com/ArcInstitute/cell-eval2/blob/main/docs/vcc2026_metrics/vcc2026-metrics-brief.md)

## 4. Experiments, tinygrad performance, and compute allocation

First establish three inexpensive baselines:

1. Resampled raw control rows with no perturbation effect and no additional count sampling.
2. Mean cell-level log-CP10k response minus mean control log-CP10k expression for each training-seen target, equally averaged across available source contexts and applied through the same count-export policy; unseen targets use no change.
3. Ridge regression on target features, context features, and explicit target–context interactions. Project ESM embeddings to at most 32 training-fitted components, use the outer product with the 64-dimensional context descriptor, and include main effects. Fit the same mean cell-level log-CP10k effects as the mean-source baseline and apply predictions as residuals through the shared export policy; select ridge strength from `{0.1, 1, 10, 100}` on source development groups. Do not fit transforms on held-out response data.

Screen four neural configurations with seed 42:

- Full adaptation with distributional invariant alignment.
- Alignment disabled.
- ESM and target baseline/detection features replaced with identifier conditioning, retaining context and control indicators. For unseen targets, use an explicit zero/unknown vector, never an untrained random target embedding; report this ablation separately on seen and unseen targets.
- Population-effect loss disabled.

Continue the two strongest configurations with seed 43. Select configurations using equal weighting of the development panels: source held-out groups, H1 development unseen targets, and RPE1 development groups. Within each panel, average ranks across the six official metrics with their documented directions; average panel ranks and break ties using source validation population-effect error. Freeze the candidate set and panel definitions before screening. Report individual metrics and coverage; these ranks are a local selection heuristic, not an official aggregate.

Use `cell-eval2==0.19.0` as the initial compatibility candidate, not an unverified permanent pin. Before screening, run the official small H1 fixture and submission validator, record the package revision, preset hash, reference-bundle version, and competition rule version, and check against the intended release. If incompatible, select the documented compatible release, rerun the fixture, and lock that exact environment before comparing models. Do not upgrade merely because a newer version exists. Report undefined/unsupported metrics as unavailable; require all six on the fixed selection panels or revise those panels before screening. [Evaluation package](https://github.com/ArcInstitute/cell-eval2), [tutorial](https://github.com/ArcInstitute/cell-eval2/blob/main/docs/tutorial.md).

Use one 4090 per run and independent processes on the second card; avoid distributed training. Pin a tested tinygrad Git SHA before the first benchmark and record the actual GPU backend, driver, and compiler. Begin with the CUDA backend on Linux and verify process/device isolation. CPU or Metal may serve local correctness checks; performance acceptance is on the rented 4090 setup.

Performance work proceeds only after numerical correctness:

- Capture fixed-shape training and inference steps with `TinyJit`. Keep data loading and export outside the captured step; pass changing minibatches and fresh noise explicitly. Test that replay does not freeze noise or training coefficients.
- Benchmark synchronized end-to-end steps after compilation/warmup, including loading, accumulation, and optimizer updates. Record compile time separately, peak memory, examples/second, and a full-run cost estimate.
- Profile before adding kernels. A fused batched triangular solve is only a candidate; decoder matrix operations, reductions, or host transfers may dominate instead. Keep the plain tinygrad implementation as a fallback.
- Any training kernel must provide correct gradients, match the float32 reference, and improve median end-to-end step time by at least 10% across three repeated runs on identical workloads. Spend at most two instance-hours of the recovery allowance on initial kernel experiments; otherwise retain standard operations. Do not promise a speedup in advance. [tinygrad documentation](https://docs.tinygrad.org/), [performance guidance](https://docs.tinygrad.org/developer/speed/).

| Stage | Two-GPU instance-hour cap |
|---|---:|
| Environment checks, feature extraction, throughput pilot | 2 |
| Configuration screening | 12 |
| Confirmation runs and reserved-target evaluation | 12 |
| Final refitting, seeds 42 and 43 | 28 |
| Full evaluation and prediction exports | 8 |
| Recovery allowance, including optional kernel experiments | 8 |
| **Total** | **70** |

At the quoted $0.60/hour combined rate, 70 instance-hours cost $42, leaving $8 for other charges or contingency within the $50 total. Verify the actual rental and storage/transfer charges before provisioning. Count compilation, evaluator work, idle time, and both concurrent jobs against shared instance time; do not double-count a simultaneous pair as two instance-hours. These are limits, not runtime predictions. Reduce experiment count before exceeding the cap, and report skipped runs.

Evaluate a 50/50 mixture of the two selected seeds' generated cells during development at the same total cell count as a single model. Use it in final prediction only if its development rank improves; freeze this before either final test. Final outputs include both checkpoints if the budget permits, baseline comparisons, coverage and stability reports, and per-context predictions.

## 5. Tests, acceptance criteria, and defaults

Before substantial GPU spending, require:

- Synthetic tests for identifier alignment, measurement masks in encoder and losses, normalization axes, unsupported genes, labels, and deterministic splits. Prove that final-test responses cannot enter training, fitted features, cached effects, or development selection.
- Reference parity for outputs, loss terms, and gradients with fixed weights/noise; float32 tolerance `atol=1e-5, rtol=1e-4` on well-conditioned fixtures. Test structural solves against a float64 NumPy solve, including `W=0`, random bounded triangular matrices, and deliberately ill-conditioned cases that trigger diagnostics. Check solve gradients with finite differences.
- Conditional Gaussian KL tests: identical distributions give zero, nonstandard priors agree with an independent calculation, and noise-space KL agrees with transformed full-covariance Gaussian KL for a fixed invariant latent on small fixtures.
- MMD tests for permutation invariance and identical empirical populations; monitor learned invariant variance and collapse in the pilot. Ensure missing genes do not affect masked loss denominators and batches of one remain finite where supported.
- Inference that runs without perturbed expression. Tests for shared target/control noise, exact identical-condition zero difference, exact null export, preserved unsupported raw counts, expected-count conservation, all-zero supported weights, and legal sampled counts.
- Export-level baseline checks: copied controls remain byte-equivalent in values/order, fallback rows match source rows, and the stochastic generation path's pseudobulk and dispersion are inspected independently of posterior reconstruction. Use the same export policy for model comparisons.
- TinyJit replay with changing inputs/noise; finite accumulated gradients; mixed-precision comparison with float32 loss and gradients on fixed fixtures before enabling it. Check checkpoint restoration of model, optimizer/master state, RNG streams, sampler position, scaler, warmup, early stopping, and budget counters. Verify uninterrupted versus resumed next-step equivalence on the pinned backend.
- One-GPU peak memory below 20 GB, a two-process isolation/concurrency check, and a synchronized throughput/cost pilot. Budget interruption must save a resumable checkpoint and stop before the cap.
- Successful official evaluator fixture and schema validation, including gene/label mismatches and excessive library totals. Reuse the pinned official metric implementation rather than reimplementing its scoring rules.

Acceptance means a reproducible tinygrad implementation, valid predictions, honest held-out results, and spending within the limit. Improved challenge performance remains an experimental outcome. If the neural adaptation loses to a baseline, retain and report it and export the stronger predictor selected on development data; do not switch predictors based on the final test.

Default deployment is Linux with Python 3.11, an exact tested tinygrad revision, and separately locked auxiliary/evaluator environments. Keep all new artifacts in the project; do not depend on `Desktop/datasets`. Read contexts and target lists from each release's manifest rather than hardcoding A/B/C. Generate and validate submission files locally; automatic leaderboard submission is excluded.
