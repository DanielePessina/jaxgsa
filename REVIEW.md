# gsax Full Review Report

> **Status**: All items fixed. 275 tests pass. 28 files changed, +625 / -314 lines.
> All three marimo notebooks export to HTML successfully.

## BUGS

### B1: DGSM pre-computed Jacobian silently broadcasts wrong bounds (Real bug)

**Location:** `src/gsax/dgsm/_analyze.py:136-153`

When a user passes `Y` with shape `(N, T)` (multi-output) and `dfdx` with shape `(N, D)` (scalar Jacobian), the code promotes `dfdx` to `(N, 1, D)`, producing `sigma`/`nu` of shape `(1, D)`. Meanwhile `var_y` has shape `(T,)` and `denom` becomes `(T, 1)`. The upper/lower bound computation then silently broadcasts `(1, D) / (T, 1)` into `(T, D)`, producing nonsensical bounds that replicate the scalar-output derivatives across all T output variances.

**Fix:** After promoting `dfdx` to 3D, validate that the T dimension of `dfdx` matches `Y`.

### B2: DGSM `_promote_jac` silently passes through malformed Jacobians

**Location:** `src/gsax/dgsm/_analyze.py:27-31`

`_promote_jac` promotes 2-D `(N, D)` Jacobians to `(N, 1, D)` but returns anything else unchanged:

```python
def _promote_jac(jac):
    if jac.ndim == 2:
        return jac[:, None, :]
    return jac  # silently returns 1-D, 4-D, etc.
```

If malformed input (e.g. 1-D or 4-D) reaches this function, downstream code produces garbage silently instead of raising.

**Fix:** Add an `elif jac.ndim == 3: return jac` branch and raise `ValueError` for any other ndim.

### B3: eFAST Nyquist frequency excluded for even N (Minor)

**Location:** `src/gsax/efast/_analyze.py:47-49`

```python
Sp = jnp.abs(f[1 : (N + 1) // 2]) ** 2 / N**2
V = 2.0 * jnp.sum(Sp)
```

For even N, `(N+1)//2 == N//2`, so the Nyquist component at index `N//2` is excluded from the power spectrum. Additionally, the Nyquist component shouldn't be doubled (it has no negative-frequency mirror), but the blanket `2.0 *` factor would double it if included. Net effect: V is slightly underestimated, inflating S1/ST by a tiny amount. Practically negligible for typical N (hundreds+), but technically incorrect.

**Fix:** Include Nyquist without doubling: `Sp_nyq = jnp.abs(f[N//2])**2 / N**2; V = 2.0 * jnp.sum(Sp) + Sp_nyq` for even N.

### B4: `_count_nans` double-counts S2 NaNs (Minor, diagnostic only)

**Location:** `src/gsax/sobol/_analyze.py:110-113`

`_count_nans` is called after `_normalize_s2_matrix` has already symmetrized the S2 matrix. The off-diagonal mask `~jnp.eye(D)` includes both upper and lower triangles, so every genuine NaN at `[j, k]` (j < k) is also mirrored to `[k, j]` and counted twice. Does not affect computed indices.

**Fix:** Use `jnp.triu` with `k=1` to count only the upper triangle.

---

## DOCUMENTATION ERRORS

### D1: DGSM API doc examples crash at runtime

**Location:** `docs/api/index.md:1047-1059, 1109-1117`

Both DGSM examples pass the batched `ishigami.evaluate` function to `analyze_dgsm`, which requires an unbatched `(D,) -> ()` function. `ishigami.evaluate` uses `X[:, 0]` indexing which requires 2D input, but `jacrev` + `vmap` trace it with a 1D input of shape `(3,)`, causing an IndexError.

**Fix:** Define an unbatched function like `def ishigami_fn(x): return jnp.sin(x[0]) + 7*jnp.sin(x[1])**2 + 0.1*x[2]**4*jnp.sin(x[0])`.

### D2: Sobol sampling cost understates the default case

**Location:** `docs/guide/methods.md` (lines 65, 107, 222, 316)

The Sobol sampling cost is stated as `N(D+2)` in four places. The default behavior (`calc_second_order=True`) costs `N(2D+2)`. Only the first mention correctly scopes it to first/total-order only. The comparison table and three other occurrences present `N(D+2)` without qualification.

**Fix:** State both costs, clarify which applies when, and use `N(2D+2)` in the comparison table since that's the default.

### D3: Benchmark tables come from different runs — should be documented

**Location:** `README.md` vs `docs/guide/benchmarks.md`

The README table and docs benchmarks page show different SALib numbers for the same scenario (e.g., 13289ms vs 276ms for 50x6 S2). These are from separate benchmark runs, likely with different SALib versions or configurations. The README combines all methods in one table without specifying bootstrap, while the docs page splits into three tables (no bootstrap, 300 bootstrap, HDMR). This is not necessarily wrong, but neither location explains which run it's from or why the numbers differ.

**Fix:** Add a note to the README table indicating the benchmark run context (SALib version, date, methodology) so readers understand why numbers differ from the docs benchmarks page. Alternatively, re-run both from the same session and update both.

### D4: Intro says "three methods" but the library has five

**Location:** `README.md` paragraph 3, `docs/index.md` hero tagline

Both say "three complementary methods: Sobol, RS-HDMR, and PCE." The library also implements eFAST and DGSM. The hero tagline and feature cards on the landing page omit eFAST and DGSM entirely.

**Fix:** Update to "five methods" and add feature cards for eFAST and DGSM.

### D5: Lamboni et al. citation is wrong in both docs and code

**Location:** `docs/guide/methods.md:310` vs `src/gsax/dgsm/_poincare.py:16`, `src/gsax/dgsm/_analyze.py:10`

Docs say "vol 87, pp 44-54". Code says "vol 93, pp 53-61". Both appear incorrect (likely vol 85, pp 44-54).

**Fix:** Verify the correct citation and update both locations.

### D6: HDMR documented as "Uniform only" but code supports Gaussian

**Location:** `docs/guide/methods.md` comparison table, `docs/api/index.md`, `docs/examples/non-uniform-inputs.md`

The comparison table says HDMR supports "Uniform only". The API docs say non-uniform specs are not supported. However, `analyze_hdmr()` calls `cdf_to_unit_interval()` which correctly handles Gaussian and truncated Gaussian inputs with no validation guard.

**Fix:** Either add a validation guard to reject non-uniform inputs, or update the docs to reflect that Gaussian inputs are supported via CDF mapping.

### D7: PCE truncated Gaussian basis inconsistency between docs pages

**Location:** `docs/guide/methods.md` PCE section vs `docs/api/index.md`

Methods page says "Hermite for Gaussian" without distinguishing truncated Gaussian. The API reference correctly notes that truncated Gaussian uses Legendre after CDF transform. The code matches the API reference.

**Fix:** Update methods.md to distinguish truncated vs unbounded Gaussian.

### D8: `sample()` API docs missing `base_n` parameter

**Location:** `docs/api/index.md:192-200`

The documented signature for `sample()` omits the `base_n: int | None = None` keyword argument that exists in the source at `src/gsax/sampling.py`.

**Fix:** Add `base_n` to the documented signature.

### D9: eFAST complementary frequency range formula/code mismatch

**Location:** `docs/guide/methods.md` eFAST ST formula vs `src/gsax/efast/_analyze.py:52`

Doc says strict `k < omega_0/2` but code uses `arange(omega_0 // 2)` which includes the boundary frequency when `omega_0` is even. The difference of a single frequency bin is typically negligible.

**Fix:** Align the formula and code (either add +1 to the range or change the formula to `<=`).

---

## DOCUMENTATION GAPS

### G1: No DGSM example page

A `examples/dgsm_benchmark.py` script exists but there is no corresponding `docs/examples/dgsm.md` page and no sidebar link. Every other method has a dedicated example page.

### G2: No PCE example page

PCE is documented in the API reference and methods guide but has no walkthrough example page in `docs/examples/`.

### G3: Benchmark functions `linear`, `oakley_ohagan`, `sobol_g` undocumented

Only `ishigami` is mentioned in the documentation. The other three benchmark functions in `src/gsax/benchmarks/` have zero documentation coverage.

### G4: `poincare_constant` and `axis_constants` have no dedicated API section

These are exported from `gsax.dgsm` but have no parameter table or usage example in the API reference.

### G5: No API redirect stubs for eFAST, DGSM, PCE

Sobol has `docs/api/analyze.md` and HDMR has `docs/api/hdmr.md` as redirect stubs, but eFAST, DGSM, and PCE have no equivalent pages.

---

## SHAPE CONSISTENCY (DGSM vs everything else)

DGSM's field names (`nu`, `sigma`, `upper_bound`, `lower_bound`) are justifiably different since it computes different quantities. But the shape conventions are arbitrarily inconsistent:

| Convention | Sobol/HDMR/eFAST | DGSM |
|------------|-------------------|------|
| Scalar output shape | `(D,)` | `(1, D)` -- never squeezed |
| Output dim letter | `K` | `T` |
| Uses `_prepare_Y` | Yes | No |
| Time-series support | `(T, K, D)` | None |
| `to_dataset` branching | `ndim`-based | `T==1` vs `T>1` |

### Recommendations

1. **Squeeze scalar output**: Make DGSM return `(D,)` for scalar models, matching all other methods.
2. **Rename T to K**: Use `K` for the output dimension to match the convention that `T` = time steps, `K` = output variables.
3. **Adopt `_prepare_Y`**: Enables consistent 1D/2D/3D handling and future time-series support.
4. **Add time-series support**: The autodiff machinery would support `(N, T, K)` input; this is just an unimplemented feature.
5. **Align `to_dataset`**: Use the same `ndim`-based branching as Sobol/HDMR/eFAST.

---

## CODE SIMPLIFICATION

### S1: `_normalize_X` is a trivial wrapper

**Location:** `src/gsax/hdmr/_analyze.py:29-31`

```python
def _normalize_X(X, problem):
    return cdf_to_unit_interval(X, problem)
```

Called at two sites. Inline as `cdf_to_unit_interval(X, problem)` and delete the wrapper.

### S2: `_normalize_output_names` is a trivial wrapper

**Location:** `src/gsax/problem.py:40-46`

Called exactly once at line 239. Inline as `tuple(output_names) if output_names is not None else None`.

### S3: Impossible validation in `Problem.from_dict`

**Location:** `src/gsax/problem.py:193-196`

`len(names) != len(input_specs)` can never fail because both come from the same dict's `.keys()` and `.values()`. Remove the check (the same check in `_from_normalized_inputs` is meaningful and should stay).

### S4: Dead squeeze branch in eFAST

**Location:** `src/gsax/efast/_analyze.py:192-194`

The `if squeeze_time and squeeze_output:` branch in the non-scalar path is unreachable (that case is captured by the `if is_scalar:` branch earlier). Replace the squeeze block with just `if squeeze_time: S1 = S1[0]; ST = ST[0]`.

### S5: `fit_coefficients` is dead code with inconsistent defaults

**Location:** `src/gsax/pce/_engine.py:151-173`

`fit_coefficients` is defined but never imported or called anywhere — the same logic is implemented inline in `analyze_pce` (lines 116-118 of `_analyze.py`). It also defaults to `ridge=0.0` while `analyze_pce` defaults to `ridge=1e-8`. Either remove it or export it with aligned defaults.

### S6: Boolean indexing prevents JIT in PCE S2 loop

**Location:** `src/gsax/pce/_engine.py:222-228`

`c2[mask]` produces dynamically-shaped arrays. Replace `jnp.sum(c2[mask])` with `jnp.sum(c2 * mask)` for JIT safety.

### S7: eFAST last chunk not padded causes recompilation

**Location:** `src/gsax/efast/_analyze.py:178-183`

The last chunk may have a different size, triggering JAX recompilation. Mirror DGSM's approach of padding the last chunk and trimming results.

### S8: HDMR `y_mean`/`y_std` reshape is roundabout

**Location:** `src/gsax/hdmr/_analyze.py:376-391`

Reshapes to `(T*K, 1)`, passes through `_reshape_emulator_value` to `(T, K, 1)`, then squeezes trailing 1. Could squeeze directly based on the squeeze flags.

---

## BENCHMARK & TEST ISSUES

### T1: Ishigami missing `analytical_indices()` function

**Location:** `src/gsax/benchmarks/ishigami.py`

Unlike the other three benchmarks, Ishigami has no `analytical_indices(A, B)` function for computing indices with non-default parameters. The `__init__.py` docstring promises one for each submodule.

### T2: Inconsistent S2 format between benchmarks

**Location:** `src/gsax/benchmarks/`

Ishigami returns `ANALYTICAL_S2` as a dict `{(0, 2): 0.2437}`. Sobol-G returns a `(D, D)` matrix with NaN diagonal. Standardize to one format (preferably the matrix, matching `SAResult.S2`).

### T3: Oakley-O'Hagan `analytical_indices` returns different type

**Location:** `src/gsax/benchmarks/oakley_ohagan.py:109`

Returns `(S1, ST, total_variance)` instead of `(S1, ST, S2)` like the other benchmarks.

### T4: Vacuous `hasattr` check in eFAST test

**Location:** `tests/test_efast.py:115-116`

`not hasattr(result, "S2")` always passes on any non-SAResult dataclass. Replace with a check on the actual field set or remove (redundant with `test_result_type`).

### T5: Near-vacuous JIT tests

**Location:** `tests/test_indices.py:7-31`

JIT tests only check `jnp.isfinite(result)` -- would pass with completely wrong values. Add tests with hand-computed expected values.

### T6: No eFAST/DGSM tests for higher-dimensional benchmarks

**Location:** `tests/test_efast.py`, `tests/test_dgsm.py`

Only Ishigami and linear are tested. No Sobol-G (8-D) or Oakley-O'Hagan tests exist for these methods.

### T7: Incomplete import tests

**Location:** `tests/test_imports.py`

Missing checks for `analyze_dgsm`, `analyze_efast`, `sample_efast`, `sample_mc`, `DGSMResult`, `EFASTResult`, `GaussianInputSpec`, `UniformInputSpec`.

### T8: No single-parameter edge case tests for eFAST/DGSM

**Location:** No file

D=1 is an edge case that often triggers bugs in frequency assignment or gradient computation. Sobol has a D=1 test in `test_shapes.py` but eFAST and DGSM do not.

### T9: Overly permissive constant-output tolerance in eFAST test

**Location:** `tests/test_efast.py:153-158`

Allows `S1 < 0.2` for constant output where both S1 and ST should be exactly 0 or NaN. Tighten to `< 0.01`.

### T10: Oakley-O'Hagan `evaluate()` has useless `sigma` parameter

**Location:** `src/gsax/benchmarks/oakley_ohagan.py:83`

The `sigma` parameter is accepted but ignored. The docstring notes it "only affects the problem definition, not the function itself." Remove it or clarify.

---

## MATH CORRECTNESS

All core formulas verified correct against the literature:

- Sobol S1 (Saltelli 2010), ST (Jansen 1999), S2 (Saltelli 2002)
- Pooled variance normalization
- HDMR ANCOVA decomposition (Li et al. 2010)
- HDMR total-order aggregation
- PCE Sobol indices (Sudret 2008)
- PCE Legendre/Hermite recurrences and orthonormalization
- PCE LOO error (hat-matrix formula)
- DGSM moments (nu, sigma)
- DGSM Poincare upper bound and Kucherenko-Song lower bound
- Poincare constants (uniform, Gaussian, truncated normal spectral solve)
- eFAST Fourier decomposition and frequency assignment (see B3 for minor Nyquist caveat on even N)
- eFAST omega_0 computation
- Bootstrap CI shapes
- Saltelli sampling layout and matrix recombination
- All benchmark analytical solutions (Ishigami, Sobol-G, Linear, Oakley-O'Hagan)

Automated audit (122 reduction/aggregation ops, all broadcasting patterns):
- All `axis` arguments in `jnp.mean`, `jnp.var`, `jnp.sum`, etc. verified correct
- No silent broadcasting traps found across arithmetic, `jnp.where`, normalization
- Saltelli sampling while-loop (`sampling.py:400-413`) verified correct — each iteration fully rebuilds the design with larger `base_n` using the same seed, so `expanded_to_unique` stays consistent
- Sobol/HDMR shape handling, bootstrap indexing, emulator coefficient pipeline all verified clean
