# Scale and limits

This page tells you how many parameters each method can take. $D$ is the
number of parameters and $N$ is the sample count.

Every method has a different limit, and the limits are far apart. Two methods
stop being usable at about 100 parameters. Three run at 5000 parameters in
under a second. Read the table before you plan a high-dimensional study.

## What each method supports

Measured on one machine. See [How these numbers were measured](#how-these-numbers-were-measured)
for the setup, and read the ceiling as an order of magnitude, not an exact
number.

| Method | Design cost in model runs | Works to | What sets the limit |
|---|---|---|---|
| Morris | $r(D+1)$ | 5000+ | Nothing yet. Analysis time is flat in $D$. |
| DGSM | $N$ Jacobians | 5000+ | Nothing yet. Cheapest method at high $D$. |
| Sobol' | $N(D+2)$, or $N(2D+2)$ with second order | 5000+ | Memory for the design array. See [The design array is the real limit](#the-design-array-is-the-real-limit). |
| Borgonovo delta | given data | 5000 | Analysis time. About 25 s at $D = 5000$. |
| Optimal transport | given data | 5000 | Analysis time. About 39 s at $D = 5000$. |
| PAWN | given data | 5000 | Analysis time. About 149 s at $D = 5000$. |
| Kucherenko | $N(2D+1)$ | 1000 | Design memory, and sampling time above $D = 1000$. |
| PCE | given data | 1000 | You need $N > 2D$ rows. The order drops to 1 above $D \approx 100$. |
| Shapley effects | given data | 1000 | Same as PCE, which is its default backend. |
| RS-HDMR | given data | 500 | `maxorder=2` builds $D^2$ component functions. |
| eFAST | $\ge 4M^2 D(D-1)$ | 100 | The design itself. See [eFAST](#efast-grows-with-the-square-of-d). |
| VKOGA | given data | 100 | Greedy centre selection. |
| HSIC | given data | 10 to 30 | $N \times N$ kernels times `n_perms`. Set `n_perms` lower to go further. |

Three of these limits move if you change a setting:

- **HSIC** costs $D \times N^2 \times$ `n_perms`. The default `n_perms=200` is
  what sets the low ceiling. Lower it, or lower $N$, and HSIC goes further.
- **RS-HDMR** costs $D^2$ at the default `maxorder=2`. Set `maxorder=1` and the
  cost becomes linear in $D$, but you lose every interaction term.
- **PCE** reduces its own order to fit your sample budget and warns when it
  does. Above $D \approx 100$ with $N = 4096$ it drops to order 1, which is a
  linear model. It then cannot see interactions at all. Give it more rows, or
  read the warning and accept main effects only.

## Pick a method for a high-dimensional problem

1. If your model is written in JAX and you can differentiate it, use
   **DGSM**. It ran 5000 parameters in 0.91 s and 501 MB from 1024 sample
   points. Nothing else comes close at high $D$.
2. If you cannot differentiate the model, screen with **Morris**. It costs
   $r(D+1)$ runs and its analysis time does not grow with $D$.
3. Fix the parameters that screening says do nothing. Then spend your budget
   on **Sobol'** for the survivors.

Do not start with Sobol' at high $D$. The next section says why.

## The design array is the real limit

The Saltelli design has $N(D+2)$ rows of $D$ columns. The array grows with the
square of $D$. Analysis is not the problem. The array is.

At $D = 5000$ with a base count of $N = 1024$:

| | Size |
|---|---|
| The design array you must hold and evaluate | 102 GB |
| The outputs `Y` | 20 MB |
| Peak memory `sobol.analyze` adds | about 0 |

`sobol.analyze` is cheap at any $D$. It ran in 0.48 s at $D = 5000$. The cost
is holding the design and running your model on it.

This also means [`jaxgsa.config.set_memory_budget`](/guide/configuration) does
not help here. That budget sizes transient arrays inside an analysis. The
design array is created by `sample()` before any analysis starts, and you hold
it, not jaxgsa.

If the design does not fit, lower the base count $N$. Watch what that costs
you: Monte Carlo error falls with the square root of $N$, so a base count of 4
is not a usable answer.

## eFAST grows with the square of D

eFAST needs `n_per_curve >= 4*M^2*(D-1) + 1` points, and it uses one curve per
parameter. The number of model runs therefore grows with $D^2$, and the design
array grows with $D^3$.

| $D$ | Model runs at $M = 4$ | Design array |
|---|---|---|
| 10 | 5770 | 0.2 MB |
| 100 | 633,700 | 236 MB |
| 500 | 15,968,500 | 29.7 GB |
| 1000 | 63,937,000 | 238 GB |

$M = 1$ is the floor and it only divides the cost by 16. eFAST at $D = 500$
still needs 1.9 GB there. Use eFAST below about 100 parameters. Above that,
pick Sobol' or Morris.

## How these numbers were measured

One sweep over $D \in \{10, 50, 100, 500, 1000, 5000\}$ for all thirteen
methods.

- Apple M1 Pro, 16 GB, CPU only, float32.
- The model was $f(x) = \sum_j x_j/(j+1) + 5 x_0 x_1$. Parameters 0 and 1
  dominate at every $D$. Every method that ran ranked both of them on top.
- One scalar output. A model with many outputs or time steps costs more.
- Given-data methods got $N = \max(4096,\ 2D)$ rows.
- A method "works to" a value of $D$ if it finished in under 300 s inside a
  600 MB design budget.

Your limits will differ. Treat the table as a guide to which methods to try
first, not as a specification.
