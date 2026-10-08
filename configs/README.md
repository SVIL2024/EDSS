# Experiment launchers

These are the two public launchers for the EDSS paper recipes. Run them from
the repository root after creating local feature manifests:

```bash
bash configs/edss_ucf.sh
bash configs/edss_xd.sh
```

Both launchers use seed 234, write fresh logs, and pass any additional options
to the corresponding Python entry point. Both run for ten epochs. UCF-Crime
evaluates every ten optimization steps; XD-Violence evaluates every fifty steps
and at the end of each epoch.

The run-best checkpoints are saved to `model/runbest_ucf.pth` and
`model/runbest_xd.pth`, and logs to `logs/`.
