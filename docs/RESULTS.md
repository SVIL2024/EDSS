# Results and claim scope

The paper reports the following single-seed, test-best results using seed 234
and the public EDSS launchers:

| Dataset | Metric | EDSS | Published VadCLIP reference | Difference |
| --- | --- | ---: | ---: | ---: |
| UCF-Crime | frame-level AUC | 89.00% | 88.02% | +0.98 pp |
| XD-Violence | frame-level AP | 85.76% | 84.51% | +1.25 pp |

The VadCLIP values are published reference numbers, not a same-environment
re-evaluation. The EDSS values use test-set model selection, so they should be
described as test-best results rather than estimates of multi-run performance.

## Controlled evidence

The completed same-environment controls provide useful context:

| Dataset | Method-off control | EDSS recipe | Difference |
| --- | ---: | ---: | ---: |
| UCF-Crime AUC | 0.874697 | 0.889970 | +0.015272 |
| XD-Violence AP | 0.848091 | 0.847282 | -0.000809 |

These controls support a positive result for UCF-Crime in the tested setting,
but not an improvement for the strict XD-Violence comparison. Fixed-budget
selector controls also reached 0.892301 AUC on UCF-Crime and 0.858326 AP on
XD-Violence, exceeding the corresponding adaptive e-BH rows in the completed
single-seed screening. Therefore the evidence does not isolate adaptive e-BH
budgeting as the source of the overall gain.

## What may be claimed

The supported description is: EDSS is an evidence-guided dense snippet
supervision objective for training weakly supervised VAD models. The current
results show dataset- and protocol-dependent improvements, especially in the
UCF-Crime control.

The project does not claim formal false-discovery-rate control, multi-seed
stability, a universal improvement over fixed budgets, or a deployment alarm
rate. The public source release omits checkpoints, logs, and downloaded
features; the numeric results should therefore be read together with the
protocol and limitations statements rather than as independently verifiable
from source alone.
