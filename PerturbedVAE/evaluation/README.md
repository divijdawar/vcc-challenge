# Evaluation

Two independent specifications for the PerturbedVAE project:

| Folder | Purpose | Authority |
| --- | --- | --- |
| [paper/](paper/README.md) | Paper benchmark and representation diagnostics | arXiv:2605.19343v1 |
| [vcc2026/](vcc2026/README.md) | Local challenge-style scoring and official export requirements | Arc's evaluator and current challenge documentation |

These folders contain documentation and machine-readable specifications. Metric
runners, source adapters, and evaluator fixture checks still need implementation.
The JSON files are project specifications, not configurations to pass directly to
an external CLI.

Keep each run's predictions, reference IDs, gene axis, split manifest, expression
scale, seeds, checkpoint hash, resolved configuration, dependency versions,
coverage, and raw results together under the project's `artifacts/evaluation/`.
Separate results by suite, protocol, context, and run ID. Record unavailable
metrics with a reason instead of converting them to zero.

The local H1 unseen-target and RPE1 joint tests in `plan.md` are project benchmarks.
They are separate from blind VCC final inference. Freeze project splits before
fitting features; inference receives controls, target IDs, features, and a
checkpoint, while scoring alone receives held-out responses.

For challenge exports, follow [vcc2026/submission.json](vcc2026/submission.json).
The current challenge contract supersedes the older plan's control-copying and
variable-cell-count submission guidance. Generic scoring inputs and official
submission files have different contracts.
