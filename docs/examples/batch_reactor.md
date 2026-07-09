# Batch Reactor (marimo notebook)

A self-contained walkthrough of Sobol global sensitivity analysis on a
batch reactor running a first-order liquid-phase reaction
$A \to B$. The rate constant $k(T,\mathrm{pH})$ combines an Arrhenius
temperature dependence with a Hill-type pH saturation curve, and the inlet
concentration $C_{A,0}$ feeds the mass balance directly. The mechanistic
model is treated as already fitted — the example focuses on variance
attribution, not estimation.

The notebook source lives at
[`examples/batch_reactor_gsa.py`](https://github.com/danielepessina/gsax/blob/master/examples/batch_reactor_gsa.py).
Run it interactively with `uv run marimo edit examples/batch_reactor_gsa.py`, or read
the rendered output below.

<iframe
  src="/gsax/notebooks/batch_reactor_gsa.html"
  style="width: 100%; height: 90vh; border: 1px solid var(--vp-c-divider); border-radius: 6px;"
  loading="lazy"
></iframe>

## What this example covers

- A three-input Sobol problem ($C_{A,0}$, $T$, $\mathrm{pH}$) defined with
  `gsax.Problem.from_dict(...)` and uniform marginals.
- A closed-form batch reactor start-up trajectory used as the model — the kind of cheap
  surrogate you would plug a fitted mechanistic predictor into.
- A single `gsax.analyze(..., num_resamples=200, key=...)` call to obtain
  $S_1$, $S_T$, $S_{ij}$ along with bootstrap 95 % confidence intervals.
- Three plots that read the indices off:
  steady-state bar chart with CI error bars, time-resolved $S_1(t)$ and
  $S_T(t)$ with shaded bootstrap envelopes, and a pairwise $S_{ij}$ heatmap.

## See also

- [Bootstrap Confidence Intervals](/examples/bootstrap) for the bare-API
  version of the same bootstrap workflow on the Ishigami benchmark.
- [Multi-Output & Time-Series](/examples/multi-output) for the shape rules
  used here when the output is `(N, T, K)`.
