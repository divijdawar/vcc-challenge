# VCC 2026 evaluations

Three sources serve different purposes:

- [Official evaluator preset](https://github.com/ArcInstitute/cell-eval2/blob/main/src/cell_eval2/configs/vcc2026.yaml)
  defines scoring settings.
- [Metric reference](https://github.com/ArcInstitute/cell-eval2/blob/main/docs/vcc2026_metrics/vcc2026-metrics-brief.md)
  and [full specification](https://github.com/ArcInstitute/cell-eval2/blob/main/docs/vcc2026_metrics/vcc2026-metrics.md)
  define the six metrics and score scaling.
- [Challenge CLI guide](https://vcc-cli-wiki.virtualcellchallenge.org/#submission-requirements-2026)
  defines the stricter upload contract.

Checked on 2026-10-30. The linked `main` branches and website are mutable.
[metrics.json](metrics.json) and [submission.json](submission.json) are reviewed
specifications, not a tested environment lock or executable evaluator config.

Reuse `cell-eval2` for metric computation. Before comparisons, run its
[H1 fixture](https://github.com/ArcInstitute/cell-eval2/blob/main/docs/tutorial.md),
then record the exact package/repository revision, preset hash, DE backend,
dependencies, and reference identity. The plan's `0.19.0` is a compatibility
candidate; fixture success must precede locking it. Pin backend and device
explicitly after checking them; avoid backend changes between candidates.

For local benchmarks, supply a real reference containing independently reserved
source controls. Validate the pinned API's handling of prediction control rows
with a fixture. This must not add controls to the upload file. Keep local scores
separate by context and fixed target panel. Unmeasured reference genes need an
explicit measured-axis policy; zero filling them does not make a benchmark valid.
Missing metrics remain unavailable with coverage counts. A local rank heuristic
from `plan.md` is not an official normalized competition score.

Challenge final inputs contain D/E/F controls and a new target list. Response
ground truth and final scores are withheld. Use the final release for inference
and format checks; use local held-out source responses for accuracy evaluation.
Read release identity, context labels, targets, and gene order from the bundle,
and keep `manifest.json` beside `pert_counts.csv`.

Export one sparse H5AD with perturbation predictions only. Use exactly 400 rows
per target/context; supplied controls remain model inputs. Validate and package
with the authenticated official CLI (version 0.3.0 or later for final phase):

```sh
vcc prep prediction.h5ad -g /path/to/final/gene_names.csv \
  --perts /path/to/final/pert_counts.csv --dry-run
vcc prep prediction.h5ad -g /path/to/final/gene_names.csv \
  --perts /path/to/final/pert_counts.csv -o prediction.vcc
```

Here `vcc` is the external `vcc-cli` executable, separate from this project's
planned `python -m vcc` interface. These commands validate/package locally and
do not submit a prediction.
