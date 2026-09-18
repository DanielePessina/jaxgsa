# PLAN-WASM: jaxgsa in the browser (client-side wasm/WebGPU)

Local planning doc (untracked, like PLAN-V1.0.md). Decides where to continue the
jaxgsa-wasm work. Source of truth for the port remains the Python package.

## Goal

A static website that runs global sensitivity analyses entirely on the visitor's
hardware (no server compute). Visitors upload model inputs/outputs or generate a
design, and the browser computes sensitivity indices via jax-js (wasm/WebGPU).

- Hosting: GitHub Pages (static, HTTPS). No custom headers -> no COOP/COEP -> no
  SharedArrayBuffer -> wasm runs single-threaded. Accepted.
- Target audience: Chrome/Edge (WebGPU everywhere) -> WebGPU is the fast path.
- Primary compute backends, in order:
  1. `webgpu` (float32) — fast path on Chrome/Edge.
  2. `wasm` (float64) — correctness path and fallback (single-threaded on Pages).
- **Precision policy (decided):** WebGPU has no float64. The WebGPU path runs in
  float32 and shows a user-facing warning that results were computed in single
  precision (same pattern as the existing single-precision warning in
  `jaxgsa.hsic`). wasm float64 remains the reference/correctness path.
- Non-Chrome/Edge (Firefox/Safari without WebGPU, older macOS): single-threaded
  wasm float64. Acceptable, may be slow for pce/hdmr fits.

## Why jax-js, not jax2onnx

- jax2onnx exports jitted JAX graphs to ONNX; kucherenko is deliberately pure
  NumPy and cannot be exported. Control flow (`lax.scan`/`while_loop` in hdmr)
  is exactly where ONNX export breaks. Static graphs don't fit configurable,
  user-facing analyses.
- jax-js is a JAX-API-compatible TS runtime that synthesizes its own wasm/WebGPU
  kernels. "Compiling to wasm" = hand-porting each method's compute core to
  `@jax-js/jax`.

## Scope

Ship six methods first; the rest (hsic, vkoga, optimal_transport, pawn,
borgonovo, efast, dgsm, sobol sampling extras) deferred.

| Method | Python profile | Port difficulty |
|---|---|---|
| kucherenko | pure NumPy (einsum/mean/var) | trivial |
| sobol | jitted kernels, static host loops over D | medium |
| morris | jitted gather-subtract-divide | easy-medium |
| pce | einsum/gram, `solve_triangular` (emulate with `solve`) | medium |
| shapley | delegates to pce/hdmr backend | easy-medium |
| hdmr | `lax.scan` + backfit `while_loop` + host `betainc` | hardest |

Two workflows, mirroring the Python API split:

- **A — generated design** (sobol, morris, kucherenko): browser samples the
  design, user downloads it, runs their model externally, uploads outputs.
- **B — given data** (pce, hdmr, shapley): user uploads arbitrary (X, Y) point
  cloud; diagnostics (`explained_variance`, `loo_rmse`) surface coverage quality.

## run_ids (Python-side precursor)

Design-based methods take Y positionally (Y[i] belongs to `samples[i]`). This is
fine in-process but breaks at file boundaries (Excel reordering, parallel
batches). Add stable row ids at the exchange boundary only:

1. `UniqueDesignSamples.row_ids` property = `np.arange(n_runs)` (derived, not
   persisted; stable across save/load/downsample/to_morris).
2. `UniqueDesignSamples.align_outputs(row_ids, Y)` — validates length, rejects
   duplicates/missing ids, returns Y in canonical row order.
3. `analyze(..., row_ids=None)` keyword on sobol/morris/kucherenko; when given,
   align before all existing handling. `indices()` stays positional-only.
4. `to_csv(path)` on the three design classes — `run_id` column + design columns.
5. The site reuses the same `run_id` join contract in TS.

Backward compatible: positional default unchanged.

## Repo layout (this clone)

```
web/                  Vite + TS + @jax-js/jax (the site artifact, web/dist)
web/src/jaxgsa/       TS mirrors, one file per Python module
goldens/              fixtures + generator (python, pinned jaxgsa tag) + vitest comparator
src/                  Python (reference + golden generator) — pin, don't drift
```

CI (GitHub Actions): regen goldens from pinned jaxgsa tag -> run TS tests in Node
(jax-js runs wasm in Node, no browser needed) -> build + deploy `web/dist` to Pages.

## Port order

0. run_ids in Python (this branch, PR upstream).
1. Phase 0 spike — Node + jax-js f64 sanity; port kucherenko end-to-end vs
   goldens; confirm device detection and the f32 warning path.
2. Phase 1 — sobol + morris (JS Sobol' sampler; two-phase UX with run_id round
   trip; statistical parity, not scipy bit-parity).
3. Phase 2 — pce + shapley(pce) given-data workflow + diagnostics.
4. Phase 3 — hdmr + shapley(hdmr); bounded-loop backfit, scalar `betainc`
   (only evaluated host-side, `hdmr/_engine.py:162`), triangular solve emulated.
5. Phase 4 — Pages deploy, device detection, perf pass, precision warning UX.

## Validation

- Tolerance + ranking agreement vs Python goldens, never bit-exact.
- Golden files generated from a pinned released jaxgsa tag (not master).
- pytest (Python) + Vitest (TS) both green in CI.

## Risks / open questions

- f32 vs f64 on WebGPU (decided: warn and proceed).
- Parity tolerance per method (fit methods: pce/hdmr more sensitive).
- jax-js control-flow gaps (hdmr backfit; no while/scan/cond).
- Refcount memory discipline in JS loops (`.ref`/`.dispose`).
- Sobol' scramble parity: statistical equivalence only.
- Single-threaded wasm perf on non-Chrome/Edge (fallback acceptable; CF Pages
  later unlocks threads via a `_headers` file if needed).
- Keep `web/` separate from the VitePress `docs/` site.
