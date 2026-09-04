# EDSS method

EDSS is a training-time auxiliary objective for the two snippet-scoring
branches already present in VadCLIP. It does not replace the original
video-level top-$k$ MIL losses.

For a branch score $s_{vt}$, the normal reference is estimated from valid
snippets in the normal videos of the current batch. With center $\mu$ and
scale $\sigma$, the standardized score is

\[
u_{vt}=\frac{s_{vt}-\mu}{\sigma},\qquad
\log e_{vt}=\eta u_{vt}-\frac{\eta^2}{2}.
\]

The implementation keeps these values in log space. For each abnormal video,
it sorts valid snippets by `log e` and applies the e-BH-inspired condition

\[
\log e_{(k)}\ \geq\ \log\!\left(\frac{n}{\alpha k}\right),
\]

where $n$ is that video's valid snippet count. The largest passing rank is
selected, subject to the training selector's minimum-rejection and maximum
fraction safeguards. Selected abnormal snippets receive a positive BCE target.

Every valid snippet in a known-normal video receives an exact negative target.
The UCF recipe can additionally supervise a fraction of the lowest-evidence
snippets in abnormal videos as confident negatives. The selector is detached,
so gradients flow through the score logits but not through the discrete target
construction.

The total loss is the original VadCLIP MIL and prompt-separation loss plus the
weighted EDSS terms:

\[
\mathcal L=\mathcal L_{\mathrm{MIL}}+\mathcal L_{\mathrm{prompt}}
 +\lambda_s(\mathcal L_P+\lambda_N\mathcal L_N+\lambda_B\mathcal L_B).
\]

The classification branch uses its binary logit for UCF-Crime. The alignment
branch uses the numerically stable anomaly margin

\[
r^{A}_{vt}=\operatorname{logsumexp}(z^{A}_{vt,1:})-z^{A}_{vt,0},
\]

which is monotone with the evaluation score $1-p(\mathrm{normal})$ and is used
for XD-Violence. At test time the standard evaluation scripts use sigmoid or
softmax probabilities and repeat each snippet score over 16 frames.

## Selectors

`ebh` is the proposed adaptive selector. `fixed_k`, `fixed_frac`, and
`original_topk` are controlled ranking baselines; `soft` is a continuous
evidence-weighted baseline; `normal_only` disables abnormal pseudo-positives
while retaining exact normal supervision. These options are exposed by the
training entry points for local ablations, while the public launchers use the
paper's EDSS settings.
