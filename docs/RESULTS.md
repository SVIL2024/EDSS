# Results

## Benchmark comparison

Frame-level results reported in the paper:

| Method | UCF-Crime AUC (%) | XD-Violence AP (%) |
| --- | ---: | ---: |
| VadCLIP | 88.02 | 84.51 |
| EDSS | 89.82 | 85.36 |
| Δ (percentage points) | +1.80 | +0.85 |

The VadCLIP row uses the published benchmark values.

## Local training comparison

The base objective and EDSS are trained with the same features, dataset
splits, and evaluation settings:

| Objective | UCF-Crime AUC (%) | XD-Violence AP (%) |
| --- | ---: | ---: |
| Base objective | 88.60 | 84.22 |
| EDSS | 89.82 | 85.36 |
| Δ (percentage points) | +1.22 | +1.14 |

The base objective combines video-level MIL and prompt separation. EDSS adds
the snippet supervision terms described in [METHOD.md](METHOD.md).
