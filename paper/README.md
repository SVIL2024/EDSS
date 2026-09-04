# EDSS manuscript

`main.tex` is an anonymous Image and Vision Computing manuscript for:

> Evidence-Guided Dense Snippet Supervision for Weakly Supervised Video
> Anomaly Detection

The paper uses the standard Elsevier `elsarticle` class and the bibliography
in `references.bib`. Build it with a TeX installation that provides
`elsarticle`:

```bash
cd paper
latexmk -pdf main.tex
```

The two checked-in PDF figures are under `paper/figures/`. Auxiliary TeX
outputs and the compiled PDF are ignored by Git. The manuscript keeps
anonymous author metadata for public review and does not include local
writing notes, downloaded reference PDFs, checkpoints, or experiment logs.

The reported EDSS values are single-seed, test-best results. The comparison
with VadCLIP uses the published reference values; it is not a same-environment
causal re-evaluation. See [`../docs/RESULTS.md`](../docs/RESULTS.md) and
[`../docs/LIMITATIONS.md`](../docs/LIMITATIONS.md).
