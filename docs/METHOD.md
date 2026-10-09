# EDSS method

EDSS adds individual snippet targets to VadCLIP's video-level MIL objective.
It supervises the visual classification branch (C) on UCF-Crime and the
vision-language alignment branch (A) on XD-Violence.

The C-branch anomaly margin $r^C_{vt}$ is its binary logit. For alignment
logits $z^A$ with the normal class at index zero, the A-branch margin is

$$
r^A_{vt}=\operatorname{logsumexp}(z^A_{vt,1:})-z^A_{vt,0}.
$$

For either branch, valid snippets from normal videos in the current batch
provide a reference mean $\mu$ and standard deviation $\sigma$. The
standardized margin and log-evidence are

$$
u_{vt}=\frac{r_{vt}-\mu}{\sigma},\qquad
\log e_{vt}=\eta u_{vt}-\frac{\eta^2}{2}.
$$

The implementation keeps these values in log space. For each abnormal video,
it sorts valid snippets by `log e` and applies the e-BH-inspired condition

$$
\log e_{(k)}\ \geq\ \log\!\left(\frac{n}{\alpha k}\right),
$$

where $n$ is that video's valid snippet count. The largest passing rank is
selected, with a minimum of one snippet and a maximum selected fraction.
Selected snippets receive a positive binary cross-entropy target.

Every valid snippet in a labeled normal video receives a negative target.
The UCF-Crime recipe also assigns negative context targets to a fraction of
the lowest-evidence snippets outside the positive set in abnormal videos.
The selector is detached; gradients flow through the margins used by the
loss, while target construction uses detached predictions.

The total loss is the original VadCLIP MIL and prompt-separation loss plus the
weighted EDSS terms:

$$
\mathcal L=\mathcal L_{\mathrm{MIL}}+\mathcal L_{\mathrm{prompt}}
 +\lambda_s(\mathcal L_P+\lambda_N\mathcal L_N+\lambda_B\mathcal L_B).
$$

Here $\mathcal L_P$, $\mathcal L_N$, and $\mathcal L_B$ supervise positive
snippets, normal-video snippets, and context snippets, respectively. The
dataset settings are specified in `configs/edss_ucf.sh` and
`configs/edss_xd.sh`.

At test time, the C-branch score is $\operatorname{sigmoid}(r^C)$ and the
A-branch score is $1-p(\mathrm{normal})=\operatorname{sigmoid}(r^A)$.
Each snippet score is repeated over its 16 frames.

## Selectors

`ebh` is the proposed adaptive selector. `fixed_k`, `fixed_frac`, and
`original_topk` are controlled ranking baselines; `soft` is a continuous
evidence-weighted baseline; `normal_only` disables abnormal pseudo-positives
while retaining exact normal supervision. These options are exposed by the
training entry points for local ablations, while the public launchers use the
paper's EDSS settings.
