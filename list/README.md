# Dataset metadata

This directory contains public split/annotation metadata and compact ground
truth arrays. Raw videos and downloaded CLIP feature arrays are not included.

## Runtime manifests

The training and test loaders expect CSV files with columns `path,label`:

| Dataset | Training manifest | Test manifest | Ground truth |
| --- | --- | --- | --- |
| UCF-Crime | `ucf_CLIP_rgb.csv` | `ucf_CLIP_rgbtest.csv` | `gt_ucf.npy` |
| XD-Violence | `xd_CLIP_rgb.csv` | `xd_CLIP_rgbtest.csv` | `gt.npy` |

The CSV files are intentionally ignored because `path` must point to feature
files on the local machine. Use `make_list_ucf.py` and `make_list_xd.py` with
an explicit `--feature-root` to generate them. The generators never insert a
repository-specific absolute path into source code.

## Included metadata

- `Anomaly_Train.txt` and `Anomaly_Test.txt` define the UCF split.
- `Temporal_Anomaly_Annotation.txt` contains UCF temporal annotations.
- `annotations.txt` and `annotations_multiclasses.txt` contain XD annotations.
- `gt*.npy` contains the frame labels, segments, and labels consumed by the
  dataset-specific evaluation scripts.

The supplied ground truth arrays are derived evaluation metadata, not raw
video or feature data. If a different dataset release or split is used,
regenerate compatible ground truth locally before evaluation.
