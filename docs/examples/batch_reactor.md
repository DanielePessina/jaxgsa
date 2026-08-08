# Batch Reactor (marimo notebook)

This notebook answers one question about a chemical reactor: which of its three
operating conditions drives the spread in conversion, and does the answer
change over the course of the batch. You end up with a ranked set of Sobol
indices with error bars, and a plot of how that ranking moves in time. A Sobol
index is the share of output variance attributed to one input.

The reactor runs a first-order liquid-phase reaction $A \to B$. Three inputs
vary: the inlet concentration $C_{A,0}$, the temperature $T$, and the
$\mathrm{pH}$. The rate constant $k(T,\mathrm{pH})$ combines an Arrhenius
temperature dependence with a Hill-type pH saturation curve. The inlet
concentration $C_{A,0}$ feeds the mass balance directly. The mechanistic model
is treated as already fitted, so the example is about variance attribution and
not about estimation.

The notebook source lives at
[`examples/batch_reactor_gsa.py`](https://github.com/danielepessina/jaxgsa/blob/master/examples/batch_reactor_gsa.py).
Run it interactively with `uv run marimo edit examples/batch_reactor_gsa.py`, or read
the rendered output below.

<iframe
  src="/jaxgsa/notebooks/batch_reactor_gsa.html"
  style="width: 100%; height: 90vh; border: 1px solid var(--vp-c-divider); border-radius: 6px;"
  loading="lazy"
></iframe>

## What this example covers

- A three-input Sobol problem ($C_{A,0}$, $T$, $\mathrm{pH}$) defined with
  `jaxgsa.Problem.from_dict(...)` and uniform marginals.
- A closed-form batch reactor start-up trajectory used as the model. It stands
  in for the fitted mechanistic predictor you would use on real work, and it is
  cheap enough to evaluate thousands of times.
- A single `jaxgsa.sobol.analyze(..., num_resamples=200, key=...)` call to obtain
  the first-order indices $S_1$, the total indices $S_T$, and the pairwise
  indices $S_{ij}$, along with bootstrap 95 % confidence intervals. The
  bootstrap resamples the data to show how far the indices move under sampling
  noise.
- Three plots that read the indices off. A bar chart of the steady-state
  indices with confidence-interval error bars ranks the inputs. Time-resolved
  $S_1(t)$ and $S_T(t)$ curves with shaded bootstrap envelopes show whether the
  ranking changes during the batch. A pairwise $S_{ij}$ heatmap shows which
  input pairs interact.

## See also

- [Bootstrap Confidence Intervals](/examples/bootstrap) for the bare-API
  version of the same bootstrap workflow on the Ishigami benchmark.
- [Multi-Output & Time-Series](/examples/multi-output) for the shape rules
  used here when the output is `(N, T, K)`.
