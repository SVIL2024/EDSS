# Reproducibility checklist

The public repository is a source release, not a dataset or checkpoint
release. Reproduction requires the UCF-Crime and/or XD-Violence data, the
released CLIP snippet features, and feature manifests whose paths are valid on
the local machine.

1. Create an environment with a compatible PyTorch build and install
   `requirements.txt`.
2. Obtain the dataset and pre-extracted CLIP features under their applicable
   licenses.
3. Generate local CSV manifests with the scripts in `list/`, using an explicit
   feature root. Keep the generated CSV files outside version control.
4. Run `python -m pytest tests -q` from the repository root.
5. Launch `bash configs/edss_ucf.sh` or `bash configs/edss_xd.sh`.
6. Evaluate a selected checkpoint with `src/ucf_test.py` or `src/xd_test.py`.

The paper recipes use seed 234 and intentionally select the best test metric
observed during training. For a fair comparison, keep the dataset split,
feature extractor, snippet length, evaluation code, training budget, and
checkpoint-selection rule fixed across methods.

Run outputs are local state and are written to ignored directories. The
training code does not modify Git metadata, create commits, upload artifacts,
or collect contributor information.
