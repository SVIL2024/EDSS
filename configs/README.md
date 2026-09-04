# Experiment launchers

These are the two public launchers for the EDSS paper recipes. Run them from
the repository root after creating local feature manifests:

```bash
bash configs/edss_ucf.sh
bash configs/edss_xd.sh
```

Both launchers use seed 234, write fresh logs, and pass any additional options
to the corresponding Python entry point. Checkpoints and logs stay in ignored
local directories.

Superseded sweeps, controls, and one-off analysis programs are intentionally
excluded from this public release. They are not required to train or evaluate
EDSS and may depend on private artifacts or machine-specific data.
