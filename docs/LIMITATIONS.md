# Limitations

## Evidence calibration

The selector is e-BH-inspired, but the learned score margins and their
estimated normal reference have not been shown to be valid e-values. The
training implementation also imposes a minimum rejection count and can cap
the selected fraction. Consequently, the method must not be described as
formal FDR-controlled anomaly localization.

## Evaluation protocol

- Reported results use one seed (`234`) and test-best checkpoint selection.
- The published VadCLIP comparison is not a same-environment causal control.
- AUC and AP measure ranking quality, not calibrated thresholds or false
  alarms per hour.
- Results cover UCF-Crime and XD-Violence with frozen CLIP ViT-B/16 features;
  transfer to other data, backbones, or feature extractors is untested.
- Scores are repeated across 16-frame snippets, limiting boundary precision.

## Method scope

EDSS uses dataset-specific branch weights, evidence levels, caps, and normal
loss weights. The hard selector is detached, so it cannot learn its own target
construction. Batch-level normal statistics can also vary with batch
composition, especially when abnormal videos contain substantial normal
context.

## Comparison boundary

The completed controls found fixed-budget selectors above the adaptive e-BH
rows on both datasets in the available single-seed screening. Dense normal
supervision helped UCF-Crime in the tested ladder but was not beneficial by
itself on XD-Violence. These observations limit the strength of any claim that
the adaptive selector or a particular auxiliary term is universally superior.
