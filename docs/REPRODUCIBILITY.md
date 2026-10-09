# Running EDSS

Run commands from the repository root. The [README](../README.md) provides
installation commands, feature and checkpoint downloads, and feature-list
generation commands for both datasets.

## Training

After installing the dependencies and creating the feature lists:

```bash
bash configs/edss_ucf.sh
bash configs/edss_xd.sh
```

Both launchers use seed `234` and ten epochs. They save the run-best models
to `model/runbest_ucf.pth` and `model/runbest_xd.pth`, with logs in `logs/`.
Additional command-line arguments can be passed after the launcher name.

## Evaluation

Pass the downloaded or locally trained checkpoint's path:

```bash
python src/ucf_test.py --model-path /path/to/ucf_checkpoint.pth
python src/xd_test.py --model-path /path/to/xd_checkpoint.pth
```

The primary benchmark scores are `AUC1` for UCF-Crime and `AP2` for
XD-Violence.

## Code checks

```bash
python -m pytest tests -q
```
