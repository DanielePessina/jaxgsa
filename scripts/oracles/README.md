# Oracle scripts

An oracle is an independent source of truth for a number a test asserts. It can
be a closed-form derivation, a published table, another library, or an R
package. This directory holds the scripts that produce those numbers.

## The rule

Oracles run **locally only**. They never ship in the package, they are not
imported by `src/jaxgsa/`, and CI never runs them. What ships is the number:
the oracle is run by hand, and its output is typed into the test as a literal.

Because of that, every oracle-derived literal in the suite carries a provenance
block in the test docstring. The block names five things:

1. the tier (T0 to T4, see `docs/adr/0001-verification-oracle-tiers.md`),
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
(SAFEpython) run in a separate process. Install them in your own local R or
Python environment; each script says at the top what it needs. The licence
rules that require the separate process are in
`docs/adr/0003-copyleft-oracles-and-licences.md`, and the inventory of which
oracles exist at which versions is in `docs/adr/0004-oracle-inventory.md`.

## Scripts

| Script | Oracle | Tier | Regenerates |
| --- | --- | --- | --- |
| `salib_delta_class_counts.py` | SALib 1.5.2 | T2 | The seven class counts in `tests/test_borgonovo.py::TestPlischkeHeuristic::test_matches_reference_class_counts` |
| `openturns_sobol_estimators.py` | OpenTURNS 1.27 | T2 | The recorded estimator literals in `tests/test_sobol_estimators.py` |
| `hdmr_direct_form.py` | Re-derived direct-form HDMR + `scipy.stats.f.ppf` | T4 | Nothing. Run by hand to cross-check `jaxgsa.hdmr.analyze` against an independent estimator |
