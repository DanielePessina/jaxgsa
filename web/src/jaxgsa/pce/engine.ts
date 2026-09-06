/**
 * PCE engine: multi-index enumeration, 1-D orthonormal basis recurrences,
 * the tensor-product design matrix, and coefficient-to-index extraction.
 *
 * Port of `src/jaxgsa/pce/_engine.py` (plus `_core/legendre.py`). The cheap
 * parts (multi-index, polynomial recurrences, masks) run as plain JS on host
 * Float64/Int32 arrays; the design matrix, the Gram solve and the
 * `c2 @ mask` index matmuls run in jax-js `np` on the wasm device in float64.
 */

import { numpy as np } from "@jax-js/jax";
import type { ProblemSpec } from "../sampling";

// ---------------------------------------------------------------------------
// Combinatorics
// ---------------------------------------------------------------------------

/** Exact integer binomial coefficient C(n, k) (Python `math.comb`). */
export function comb(n: number, k: number): number {
  if (!Number.isInteger(n) || !Number.isInteger(k) || k < 0 || k > n) {
    throw new Error(`comb(${n}, ${k}): n and k must be integers with 0 <= k <= n`);
  }
  k = Math.min(k, n - k);
  let r = 1;
  for (let i = 1; i <= k; i++) {
    r = (r * (n - k + i)) / i;
  }
  return Math.round(r);
}

// ---------------------------------------------------------------------------
// Multi-index (`build_multi_index`, _engine.py:86)
// ---------------------------------------------------------------------------

export interface MultiIndex {
  /** (n_terms, D) int32 multi-index in row-major flat layout; row 0 is the all-zero constant term. */
  flat: Int32Array;
  nTerms: number;
  D: number;
}

/** Every tuple of `nParts` positive integers summing to `total`. */
function positiveCompositions(
  total: number,
  nParts: number,
  prefix: number[],
  out: number[][],
): void {
  if (nParts === 1) {
    out.push([...prefix, total]);
    return;
  }
  for (let v = 1; v <= total - nParts + 1; v++) {
    prefix.push(v);
    positiveCompositions(total - v, nParts - 1, prefix, out);
    prefix.pop();
  }
}

/** Every k-subset of {start, ..., n-1}, lexicographically. */
function combinationsRange(n: number, k: number, start: number, prefix: number[], out: number[][]): void {
  if (prefix.length === k) {
    out.push([...prefix]);
    return;
  }
  for (let i = start; i <= n - (k - prefix.length); i++) {
    prefix.push(i);
    combinationsRange(n, k, i + 1, prefix, out);
    prefix.pop();
  }
}

function lexCompare(a: number[], b: number[]): number {
  for (let d = 0; d < a.length; d++) {
    if (a[d] !== b[d]) return a[d] - b[d];
  }
  return 0;
}

/**
 * All alpha in N^D with |alpha| <= p, ordered by total degree then
 * lexicographically within each degree block. Row 0 is the all-zero constant
 * index; n_terms = C(D + p, p). Mirrors `build_multi_index` exactly: the
 * Python enumerates by support (positions + positive compositions) and
 * lexsorts each degree block; sorting every row by (total degree, lex) yields
 * the same order because the blocks are already degree-ascending.
 */
export function buildMultiIndex(D: number, p: number): MultiIndex {
  if (!Number.isInteger(D) || D < 1) throw new Error(`buildMultiIndex: D must be a positive integer, got ${D}`);
  if (!Number.isInteger(p) || p < 0) throw new Error(`buildMultiIndex: p must be a non-negative integer, got ${p}`);
  const nTerms = comb(D + p, p);
  const rows: { degree: number; values: number[] }[] = [];
  for (let degree = 1; degree <= p; degree++) {
    for (let nNonzero = 1; nNonzero <= Math.min(degree, D); nNonzero++) {
      const positions: number[][] = [];
      combinationsRange(D, nNonzero, 0, [], positions);
      for (const pos of positions) {
        const parts: number[][] = [];
        positiveCompositions(degree, nNonzero, [], parts);
        for (const part of parts) {
          const values = new Array<number>(D).fill(0);
          for (let k = 0; k < nNonzero; k++) values[pos[k]] = part[k];
          rows.push({ degree, values });
        }
      }
    }
  }
  rows.sort((a, b) => a.degree - b.degree || lexCompare(a.values, b.values));

  const flat = new Int32Array(nTerms * D);
  for (let t = 0; t < rows.length; t++) {
    for (let d = 0; d < D; d++) flat[(t + 1) * D + d] = rows[t].values[d];
  }
  if (rows.length + 1 !== nTerms) {
    throw new Error(`buildMultiIndex: built ${rows.length + 1} terms, expected ${nTerms}`);
  }
  return { flat, nTerms, D };
}

/** The D columns of a multi-index, each an Int32Array of length n_terms. */
export function multiIndexColumns(mi: MultiIndex): Int32Array[] {
  const cols: Int32Array[] = [];
  for (let d = 0; d < mi.D; d++) {
    const col = new Int32Array(mi.nTerms);
    for (let t = 0; t < mi.nTerms; t++) col[t] = mi.flat[t * mi.D + d];
    cols.push(col);
  }
  return cols;
}

// ---------------------------------------------------------------------------
// 1-D orthonormal bases (recurrences, plain JS)
// ---------------------------------------------------------------------------

/**
 * Orthonormal Legendre polynomials on [-1, 1] via Bonnet's recurrence,
 * degree k scaled by sqrt(2k + 1) (`_core/legendre.py`). Returns the
 * (N, max_degree + 1) table in row-major layout.
 */
export function legendreOrthonormal(x: Float64Array, maxDegree: number): Float64Array {
  const N = x.length;
  const cols = maxDegree + 1;
  const out = new Float64Array(N * cols);
  for (let i = 0; i < N; i++) out[i * cols] = 1;
  if (maxDegree >= 1) {
    for (let i = 0; i < N; i++) out[i * cols + 1] = x[i];
  }
  for (let n = 1; n < maxDegree; n++) {
    // P_{n+1}(x) = ((2n+1) x P_n(x) - n P_{n-1}(x)) / (n+1)
    const a = 2 * n + 1;
    for (let i = 0; i < N; i++) {
      out[i * cols + (n + 1)] =
        (a * x[i] * out[i * cols + n] - n * out[i * cols + (n - 1)]) / (n + 1);
    }
  }
  for (let k = 0; k <= maxDegree; k++) {
    const s = Math.sqrt(2 * k + 1);
    for (let i = 0; i < N; i++) out[i * cols + k] *= s;
  }
  return out;
}

/**
 * Orthonormal probabilist's Hermite polynomials, He_{n+1} = x He_n - n He_{n-1},
 * degree k scaled by 1/sqrt(k!) (`_engine.py:30`). Returns the (N, max_degree + 1)
 * table in row-major layout.
 */
export function hermiteOrthonormal(x: Float64Array, maxDegree: number): Float64Array {
  const N = x.length;
  const cols = maxDegree + 1;
  const out = new Float64Array(N * cols);
  for (let i = 0; i < N; i++) out[i * cols] = 1;
  if (maxDegree >= 1) {
    for (let i = 0; i < N; i++) out[i * cols + 1] = x[i];
  }
  for (let n = 1; n < maxDegree; n++) {
    for (let i = 0; i < N; i++) {
      out[i * cols + (n + 1)] = x[i] * out[i * cols + n] - n * out[i * cols + (n - 1)];
    }
  }
  let fact = 1;
  for (let k = 1; k <= maxDegree; k++) {
    fact *= k;
    const s = 1 / Math.sqrt(fact);
    for (let i = 0; i < N; i++) out[i * cols + k] *= s;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Order selection and reference mapping (`_analyze.py`)
// ---------------------------------------------------------------------------

/**
 * Reduce polynomial order until C(D + p, p) <= fit_ratio * N, never below 1
 * (`_auto_order`).
 */
export function autoOrder(D: number, N: number, maxOrder: number, fitRatio = 0.5): number {
  const cap = Math.max(1, Math.floor(fitRatio * N));
  let order = maxOrder;
  while (order >= 1 && comb(D + order, order) > cap) {
    order -= 1;
  }
  return Math.max(order, 1);
}

/**
 * Map physical inputs to the Wiener-Askey reference domain (`_map_to_reference`):
 * uniform -> [-1, 1] (Legendre), untruncated gaussian -> standardized (Hermite).
 * Truncated-gaussian and categorical marginals are not ported yet and raise.
 * `x` is (N, D) row-major.
 */
export function mapToReference(
  x: Float64Array,
  problem: ProblemSpec,
): { xRef: Float64Array; inputTypes: ("uniform" | "gaussian")[] } {
  const D = problem.names.length;
  const N = x.length / D;
  if (!Number.isInteger(N)) {
    throw new Error(`pce: x length ${x.length} is not a multiple of D=${D}`);
  }
  const xRef = new Float64Array(x.length);
  const inputTypes: ("uniform" | "gaussian")[] = [];
  for (let d = 0; d < D; d++) {
    const spec = problem.marginals[d];
    if (spec.kind === "categorical") {
      throw new Error(
        `pce: parameter "${problem.names[d]}" is categorical; no orthogonal ` +
          "polynomial family exists for an unordered marginal",
      );
    }
    if (spec.kind === "uniform") {
      inputTypes.push("uniform");
      for (let i = 0; i < N; i++) {
        xRef[i * D + d] = (2 * (x[i * D + d] - spec.low)) / (spec.high - spec.low) - 1;
      }
    } else {
      if (spec.low !== undefined || spec.high !== undefined) {
        throw new Error(
          `pce: parameter "${problem.names[d]}" is a truncated gaussian; ` +
            "truncated-gaussian marginals are not yet ported " +
            "(only uniform and untruncated gaussian are implemented)",
        );
      }
      inputTypes.push("gaussian");
      const std = Math.sqrt(spec.variance);
      for (let i = 0; i < N; i++) {
        xRef[i * D + d] = (x[i * D + d] - spec.mean) / std;
      }
    }
  }
  return { xRef, inputTypes };
}

// ---------------------------------------------------------------------------
// Design matrix (`build_design_matrix`, _engine.py:134)
// ---------------------------------------------------------------------------

/**
 * Tensor-product design matrix Phi, shape (N, n_terms): the 1-D basis tables
 * are evaluated per dimension (Legendre for uniform, Hermite for gaussian),
 * then Phi[n, alpha] = prod_d basis_d[n, alpha_d] as a running product of
 * column gathers. Runs in jax-js `np` in float64.
 */
export function buildDesignMatrix(
  xRef: Float64Array,
  mi: MultiIndex,
  inputTypes: readonly ("uniform" | "gaussian")[],
  maxDegree: number,
): np.Array {
  const D = mi.D;
  const N = xRef.length / D;
  const nTerms = mi.nTerms;

  const basisTables: Float64Array[] = [];
  const miCols: Int32Array[] = [];
  for (let d = 0; d < D; d++) {
    const col = new Float64Array(N);
    for (let i = 0; i < N; i++) col[i] = xRef[i * D + d];
    basisTables.push(
      inputTypes[d] === "uniform"
        ? legendreOrthonormal(col, maxDegree)
        : hermiteOrthonormal(col, maxDegree),
    );
    const mc = new Int32Array(nTerms);
    for (let t = 0; t < nTerms; t++) mc[t] = mi.flat[t * D + d];
    miCols.push(mc);
  }

  let Phi = np.take(
    np.array(basisTables[0] as Float64Array<ArrayBuffer>, { dtype: np.float64 }).reshape([
      N,
      maxDegree + 1,
    ]),
    np.array(miCols[0] as Int32Array<ArrayBuffer>, { dtype: np.int32 }),
    1,
  );
  for (let d = 1; d < D; d++) {
    const factor = np.take(
      np.array(basisTables[d] as Float64Array<ArrayBuffer>, { dtype: np.float64 }).reshape([
        N,
        maxDegree + 1,
      ]),
      np.array(miCols[d] as Int32Array<ArrayBuffer>, { dtype: np.int32 }),
      1,
    );
    Phi = np.multiply(Phi.ref, factor);
  }
  return Phi;
}

// ---------------------------------------------------------------------------
// Coefficient extraction (`sobol_from_coefficients`, _engine.py:176)
// ---------------------------------------------------------------------------

/**
 * Sobol indices from PCE coefficients (Sudret 2008). c2 = coeffs^2 runs in np;
 * the active / only-singleton masks are derived host-side from the multi-index
 * and the `c2 @ mask` matmuls run in np. `coeffs` is consumed (moved in); pass
 * `.ref` when the caller still needs it. S2 is not ported yet.
 */
export function sobolFromCoefficients(
  coeffs: np.Array,
  mi: MultiIndex,
): { S1: Float64Array; ST: Float64Array } {
  const D = mi.D;
  const nTerms = mi.nTerms;

  // c2 = coeffs**2; total_var = sum(c2[1:]); inv_var = 1/total_var (NaN if 0).
  const c2 = np.square(coeffs); // (n_terms,) — coeffs consumed
  const c2NonConst = c2.ref.slice([1]); // (n_terms - 1,) — slice consumes the ref, c2 stays alive
  const totalVar = np.sum(c2NonConst, 0); // 0-d — c2NonConst consumed
  const isZero = np.equal(totalVar.ref, 0); // totalVar ref'd (two uses)
  const invVar = np.where(isZero, np.nan, np.divide(1, totalVar)); // isZero, totalVar consumed

  // "Active" means variable d has nonzero degree in multi-index alpha.
  const active = new Float64Array(nTerms * D);
  const activeCount = new Int32Array(nTerms);
  for (let t = 0; t < nTerms; t++) {
    let cnt = 0;
    for (let d = 0; d < D; d++) {
      if (mi.flat[t * D + d] > 0) {
        active[t * D + d] = 1;
        cnt++;
      }
    }
    activeCount[t] = cnt;
  }
  const onlyI = new Float64Array(nTerms * D);
  for (let t = 0; t < nTerms; t++) {
    if (activeCount[t] === 1) {
      for (let d = 0; d < D; d++) onlyI[t * D + d] = active[t * D + d];
    }
  }

  const onlyNp = np.array(onlyI as Float64Array<ArrayBuffer>, { dtype: np.float64 }).reshape([
    nTerms,
    D,
  ]); // (n_terms, D)
  const activeNp = np.array(active as Float64Array<ArrayBuffer>, { dtype: np.float64 }).reshape([
    nTerms,
    D,
  ]); // (n_terms, D)
  const S1raw = np.matmul(c2.ref, onlyNp); // (D,) — onlyNp consumed
  const STraw = np.matmul(c2, activeNp); // (D,) — c2, activeNp consumed

  const S1 = np.multiply(S1raw, invVar.ref); // (D,) — S1raw consumed
  const ST = np.multiply(STraw, invVar); // (D,) — STraw, invVar consumed

  return {
    S1: S1.dataSync() as Float64Array,
    ST: ST.dataSync() as Float64Array,
  };
}