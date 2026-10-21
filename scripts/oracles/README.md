# Oracle scripts

An oracle is an independent source of truth for a number a test asserts. It can
be a closed-form derivation, a published table, another library, or an R
package. This directory holds the scripts that produce those numbers.

## The rule

Oracles run **locally only**. They never ship in the package, they are not
imported by `src/jaxgsa/`, and CI never runs them. What ships is the number:
the oracle is run by hand, and its output is typed into the test as a literal.
The one exception is `external_checks.py`, which compares jaxgsa with live
libraries on demand. It is outside the default test collection.

Because of that, every oracle-derived literal in the suite carries a provenance
block in the test docstring. The block names five things:

1. the tier (T0 closed form, T1 recorded external literal, T2 external library
   at test time, T4 internal consistency only),
2. the oracle,
3. its exact version,
4. the date it was run,
5. the path to the script in this directory that regenerates it.

A number with no provenance block is not an oracle, and the verification review
rejects it.

## What a script here must do

- Print a readable table of the values it computed.
- Compare them against the literals that are in the test today.
- Exit `0` when they agree and non-zero when they do not.

That way running the script is a one-command answer to "is this literal still
what the oracle says?".

## Running them

The oracle libraries are development extras, not runtime dependencies, so run
the Python scripts through the dev extra:

```
uv run --extra dev scripts/oracles/<script>.py
```

R oracles (`sensitivity`, `sensobol`, `gsaot`) and copyleft Python oracles
(SAFEpython) run in a separate process, because their licences are
incompatible with jaxgsa's and must never be imported into it. Install them
in your own local R or Python environment; each script says at the top what
it needs.

## Scripts

| Script | Oracle | Tier | Regenerates |
| --- | --- | --- | --- |
| `salib_delta_class_counts.py` | SALib 1.5.2 | T2 | The seven class counts in `tests/test_borgonovo.py::TestPlischkeHeuristic::test_matches_reference_class_counts` |
| `openturns_sobol_estimators.py` | OpenTURNS 1.27 | T2 | The `OPENTURNS_ISHIGAMI` literals in `tests/test_sobol_estimators.py`, used by `TestOpenTurnsOracle` |
| `hdmr_direct_form.py` | Re-derived direct-form HDMR + `scipy.stats.f.ppf` | T4 | Nothing. Run by hand to cross-check `jaxgsa.hdmr.analyze` against an independent estimator |
| `external_checks.py` | SALib, POT, SciPy (live) | T2 | Nothing. A pytest file of live comparisons, see below |

## Live checks: `external_checks.py`

`external_checks.py` holds the tests that import an outside library and compare
jaxgsa with it at run time. It covers:

- SALib Sobol parity for the default `saltelli-jansen` estimator.
- SALib Morris parity on the same expanded design.
- SALib `delta`: the plug-in estimator and the bias-corrected end-to-end values.
- PAWN against SALib's median, and against `scipy.stats.ks_2samp` on tied outputs.
- Optimal transport against POT: the 1-D W2 semantics, per-bin W2, the Sinkhorn
  solver, the multivariate pipeline against exact `emd2`, the Ishigami `x1`
  index and the quantile-rank split.

The file name does not match `test_*.py`, so `uv run pytest` does not collect
it. CI does not run it. Run it by hand:

```
uv run --extra dev pytest scripts/oracles/external_checks.py
```

It takes about 20 seconds. You can also run it as a script:
`uv run --extra dev python scripts/oracles/external_checks.py`.
