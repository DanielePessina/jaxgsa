import { sobolSequence } from "./sampler";

/**
 * Port of `jaxgsa.sobol.sample` and its helpers
 * (`_build_expanded_samples`, `_stable_unique_rows`, `_transform_samples`)
 * to plain TS. No jax-js involved: the Python sampler is host numpy and the
 * sampling step is cheap; wasm/jax-js compute comes later in analyze.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type MarginalSpec =
  | { kind: "uniform"; low: number; high: number }
  | {
      kind: "gaussian";
      mean: number;
      variance: number;
      low?: number;
      high?: number;
    }
  // Categorical marginals are NOT yet ported: the level transform raises.
  | { kind: "categorical"; probs: number[] };

export interface ProblemSpec {
  names: string[];
  marginals: MarginalSpec[];
  /** Declared dependence structure -> the Saltelli design is invalid. */
  correlation?: unknown;
}

export interface SobolDesign {
  /** Unique rows in physical units, shape (nRuns, nParams), row-major. */
  samples: Float64Array;
  /** Rows of the expanded Saltelli design before deduplication. */
  nExpanded: number;
  /** Map from each expanded row to its retained unique row. */
  expandedToUnique: Int32Array;
  baseN: number;
  nParams: number;
  calcSecondOrder: boolean;
}

// ---------------------------------------------------------------------------
// Marginal transforms (`_transform_samples`)
// ---------------------------------------------------------------------------

/** Unit-cube coordinates are clipped before the inverse-CDF (Python: UNIT_CLIP = 1e-12). */
const UNIT_CLIP = 1e-12;

/**
 * Inverse standard-normal CDF, Acklam's rational algorithm (public domain,
 * ~1e-9 absolute accuracy — plenty for inverse-CDF sampling; scipy's ndtri
 * is machine-precision but bit parity with scipy is not required here).
 * `p` must lie in (0, 1).
 */
export function normalInverseCdf(p: number): number {
  const A = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
  ];
  const B = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const C = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783,
  ];
  const D = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416,
  ];
  const P_LOW = 0.02425;
  const P_HIGH = 1 - P_LOW;

  let q: number;
  let x: number;
  if (p < P_LOW) {
    q = Math.sqrt(-2 * Math.log(p));
    x =
      (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) /
      ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1);
  } else if (p <= P_HIGH) {
    q = p - 0.5;
    const r = q * q;
    x =
      (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) *
      q /
      (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1);
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    x =
      -((((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])) /
      ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1);
  }
  return x;
}

/**
 * Complementary error function via the standard Numerical-Recipes rational
 * approximation (~1.2e-7 absolute error). Used only to rescale unit values
 * into the truncation window of a truncated Gaussian, where that accuracy
 * is ample.
 */
export function erfComplement(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r = t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t *
                          (0.27886807 +
                            t *
                              (-1.13520398 +
                                t *
                                  (1.48851587 +
                                    t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? r : 2 - r;
}

/** Standard-normal CDF Phi(x). */
export function normalCdf(x: number): number {
  return 0.5 * erfComplement(-x / Math.SQRT2);
}

/**
 * Inverse CDF of the standard-normal distribution truncated to [a, b]
 * (a, b in standard units, either may be -Infinity/+Infinity), the same
 * "rescale the unit value into the probability window" construction scipy's
 * truncnorm.ppf and the jaxgsa differentiable transform use. Windows sitting
 * in the upper tail are reflected to the survival side
 * (ppf(u; a, b) = -ppf(1-u; -b, -a)) so the window width stays numerically
 * well-conditioned; this mirrors `_jax_transform_gaussian` in
 * src/jaxgsa/_core/sampling.py.
 */
export function truncatedGaussianInverse(u: number, a: number, b: number): number {
  if (a === -Infinity && b === Infinity) return normalInverseCdf(u);
  const reflect = b === Infinity || (a !== -Infinity && a + b > 0);
  const uu = reflect ? 1 - u : u;
  const lo = reflect ? (b === Infinity ? -Infinity : -b) : a;
  const hi = reflect ? (a === -Infinity ? Infinity : -a) : b;
  const pa = lo === -Infinity ? 0 : normalCdf(lo);
  const pb = hi === Infinity ? 1 : normalCdf(hi);
  const x = normalInverseCdf(pa + uu * (pb - pa));
  return reflect ? -x : x;
}

/** Transform one unit-cube value into one marginal's physical units. */
export function transformValue(u: number, spec: MarginalSpec): number {
  if (spec.kind === "uniform") {
    // Factor order matches the Python `_transform_uniform` so the two agree
    // bitwise for identical input.
    return u * (spec.high - spec.low) + spec.low;
  }
  if (spec.kind === "categorical") {
    throw new Error(
      "jaxgsa.sobol.sample: categorical marginals are not yet ported " +
        "(only uniform and gaussian are implemented)",
    );
  }
  const clipped = Math.min(Math.max(u, UNIT_CLIP), 1 - UNIT_CLIP);
  const std = Math.sqrt(spec.variance);
  const a = spec.low === undefined ? -Infinity : (spec.low - spec.mean) / std;
  const b = spec.high === undefined ? Infinity : (spec.high - spec.mean) / std;
  let v = spec.mean + std * truncatedGaussianInverse(clipped, a, b);
  if (spec.low !== undefined) v = Math.max(v, spec.low);
  if (spec.high !== undefined) v = Math.min(v, spec.high);
  return v;
}

/** Column-wise inverse-CDF transform of the unit-cube design (`_transform_samples`). */
export function transformSamples(problem: ProblemSpec, unit: Float64Array): Float64Array {
  const D = problem.names.length;
  const nRows = unit.length / D;
  const out = new Float64Array(unit.length);
  for (let j = 0; j < D; j++) {
    const spec = problem.marginals[j];
    for (let r = 0; r < nRows; r++) {
      out[r * D + j] = transformValue(unit[r * D + j], spec);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Saltelli layout (`_saltelli_step`, `_build_expanded_samples`)
// ---------------------------------------------------------------------------

/** Rows of the expanded Saltelli design per base Sobol' point. */
export function saltelliStep(nParams: number, calcSecondOrder: boolean): number {
  return calcSecondOrder ? 2 * nParams + 2 : nParams + 2;
}

export interface ExpandedOptions {
  calcSecondOrder: boolean;
  scramble: boolean;
  seed: number;
}

/**
 * Draw `baseN` points from a 2D-dimensional Sobol' sequence (first D columns
 * = matrix A, last D = matrix B) and build the Saltelli interleaved layout:
 * per base point i the rows are
 *   [A_i, AB_0, ..., AB_{D-1}, BA_0, ..., BA_{D-1}, B_i]
 * with AB_j = A with column j replaced by B's column j (and BA_j the
 * transpose), the BA block present only for second order. Output is
 * (baseN * step, D) row-major. Exact duplicate rows from the Saltelli
 * collapse are kept here and removed later by deduplication.
 */
export function buildExpandedSamples(
  nParams: number,
  baseN: number,
  options: ExpandedOptions,
): Float64Array {
  const D = nParams;
  const step = saltelliStep(D, options.calcSecondOrder);
  const base = sobolSequence(2 * D, baseN, options.scramble, options.seed);
  const out = new Float64Array(baseN * step * D);
  for (let i = 0; i < baseN; i++) {
    const b = i * 2 * D;
    const o = i * step * D;
    // A_i
    for (let j = 0; j < D; j++) out[o + j] = base[b + j];
    // AB_0..AB_{D-1}
    for (let jj = 0; jj < D; jj++) {
      const oo = o + (1 + jj) * D;
      for (let j = 0; j < D; j++) out[oo + j] = base[b + j];
      out[oo + jj] = base[b + D + jj];
    }
    // BA_0..BA_{D-1} (second order only)
    if (options.calcSecondOrder) {
      for (let jj = 0; jj < D; jj++) {
        const oo = o + (1 + D + jj) * D;
        for (let j = 0; j < D; j++) out[oo + j] = base[b + D + j];
        out[oo + jj] = base[b + jj];
      }
    }
    // B_i
    for (let j = 0; j < D; j++) out[o + (step - 1) * D + j] = base[b + D + j];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Deduplication (`_stable_unique_rows`)
// ---------------------------------------------------------------------------

/**
 * Deduplicate rows with first-occurrence ordering and bitwise row equality:
 * two rows merge only when every float64 element is bit-identical
 * (so 0.0 and -0.0 stay distinct and equal NaN bit patterns still merge),
 * matching `_stable_unique_rows` in src/jaxgsa/_core/sampling.py. Returns
 * the unique rows in first-occurrence order and the map from each original
 * row to its unique row index.
 *
 * Keyed by a 32-bit hash of the raw row bytes with exact bitwise
 * verification on hash collision, so the semantics are exact at any size.
 */
export function stableUniqueRows(
  samples: Float64Array,
  dim: number,
): { unique: Float64Array; expandedToUnique: Int32Array } {
  const nRows = dim === 0 ? 0 : samples.length / dim;
  if (nRows === 0) {
    return { unique: new Float64Array(0), expandedToUnique: new Int32Array(0) };
  }
  const dv = new DataView(samples.buffer, samples.byteOffset, samples.byteLength);
  const nWords = Math.floor(dv.byteLength / 4);
  const words = new Uint32Array(samples.buffer, samples.byteOffset, nWords);
  const rowWords = dim * 2;

  const seen = new Map<number, number[]>();
  const uniqueRows: number[] = [];
  const expandedToUnique = new Int32Array(nRows);

  for (let r = 0; r < nRows; r++) {
    const w0 = r * rowWords;
    let h = 0x811c9dc5;
    for (let w = 0; w < rowWords; w++) {
      h = Math.imul(h ^ words[w0 + w], 0x01000193) >>> 0;
    }
    h ^= h >>> 16;
    h = Math.imul(h, 0x7feb352d) >>> 0;
    h ^= h >>> 15;
    h = Math.imul(h, 0x846ca68b) >>> 0;
    h ^= h >>> 16;

    const bucket = seen.get(h);
    let match = -1;
    if (bucket !== undefined) {
      for (const u of bucket) {
        if (rowsBitwiseEqual(dv, uniqueRows[u], r, dim)) {
          match = u;
          break;
        }
      }
    }
    if (match >= 0) {
      expandedToUnique[r] = match;
    } else {
      const idx = uniqueRows.length;
      uniqueRows.push(r);
      expandedToUnique[r] = idx;
      if (bucket === undefined) seen.set(h, [idx]);
      else bucket.push(idx);
    }
  }

  const unique = new Float64Array(uniqueRows.length * dim);
  for (let k = 0; k < uniqueRows.length; k++) {
    const src = uniqueRows[k] * dim;
    for (let j = 0; j < dim; j++) unique[k * dim + j] = samples[src + j];
  }
  return { unique, expandedToUnique };
}

function rowsBitwiseEqual(dv: DataView, a: number, b: number, dim: number): boolean {
  const rowBytes = dim * 8;
  const ao = a * rowBytes;
  const bo = b * rowBytes;
  for (let j = 0; j < dim; j++) {
    if (dv.getBigUint64(ao + j * 8) !== dv.getBigUint64(bo + j * 8)) return false;
  }
  return true;
}

/** Deduplicate on the unit cube first, then transform (`_dedupe_design`). */
export function dedupeDesign(
  problem: ProblemSpec,
  expandedUnit: Float64Array,
): { samples: Float64Array; expandedToUnique: Int32Array; nExpanded: number } {
  const D = problem.names.length;
  const { unique, expandedToUnique } = stableUniqueRows(expandedUnit, D);
  const samples = transformSamples(problem, unique);
  return { samples, expandedToUnique, nExpanded: expandedUnit.length / D };
}

// ---------------------------------------------------------------------------
// Public entry point (`sample`)
// ---------------------------------------------------------------------------

export const MAX_BASE_N = 1 << 24;

function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

function nextPowerOfTwo(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

export interface SampleOptions {
  /** Exact Sobol' base size; must be a power of 2. */
  baseN?: number;
  /** Include the BA cross-matrices needed for second-order indices (default true). */
  calcSecondOrder?: boolean;
  /** Apply the seeded affine scramble (default true). */
  scramble?: boolean;
  seed?: number;
  verbose?: boolean;
}

/**
 * Generate the input samples needed for Sobol' analysis: a Saltelli design
 * over a candidate `baseN`, deduplicated to unique rows in first-occurrence
 * order, doubling `baseN` until the requested number of unique rows exists.
 * Mirrors `jaxgsa.sobol.sample` (continuous problems only for now).
 */
export function sample(
  problem: ProblemSpec,
  nSamples: number,
  options: SampleOptions = {},
): SobolDesign {
  const {
    baseN: baseNGiven,
    calcSecondOrder = true,
    scramble = true,
    seed = 0,
    verbose = true,
  } = options;

  if (problem.correlation != null) {
    throw new Error(
      "jaxgsa.sobol.sample: correlated problems are not supported (the " +
        "Saltelli design and its estimators assume independent inputs)",
    );
  }
  const D = problem.names.length;
  if (D < 1) {
    throw new Error("jaxgsa.sobol.sample: problem must declare at least one parameter");
  }
  if (D > 20) {
    throw new Error(
      `jaxgsa.sobol.sample: at most 20 parameters supported (a Saltelli design ` +
        `needs 2D = ${2 * D} Sobol' dimensions, capped at 40)`,
    );
  }
  if (problem.marginals.length !== D) {
    throw new Error(
      `jaxgsa.sobol.sample: problem declares ${D} names but ${problem.marginals.length} marginals`,
    );
  }

  const step = saltelliStep(D, calcSecondOrder);
  let baseN: number;
  let targetN: number | null;
  if (baseNGiven !== undefined) {
    if (!isPowerOfTwo(baseNGiven)) {
      throw new Error(
        `jaxgsa.sobol.sample: base_n must be a power of 2 (got ${baseNGiven})`,
      );
    }
    baseN = baseNGiven;
    targetN = null;
  } else {
    targetN = Math.max(1, nSamples);
    baseN = nextPowerOfTwo(Math.ceil(targetN / step));
    if (baseN > MAX_BASE_N) {
      throw new Error(
        `jaxgsa.sobol.sample: n_samples=${targetN} needs base_n=${baseN}, above ` +
          `the cap of 2^24 = ${MAX_BASE_N}`,
      );
    }
  }

  if (baseN < 16) {
    console.warn(
      `jaxgsa.sobol.sample: base_n=${baseN} is below the floor of 16 base ` +
        "points. The design is degenerate: the Sobol' indices will be " +
        "unreliable. Raise n_samples (or base_n) for a usable design.",
    );
  }

  const build = () =>
    dedupeDesign(problem, buildExpandedSamples(D, baseN, { calcSecondOrder, scramble, seed }));

  let design = build();
  let nExpanded = design.nExpanded;

  if (targetN !== null) {
    let doublings = 0;
    while (design.samples.length / D < targetN) {
      if (doublings >= 32 || baseN >= MAX_BASE_N) {
        console.warn(
          `jaxgsa.sobol.sample: n_samples=${targetN} unique rows cannot be ` +
            `reached${baseN >= MAX_BASE_N ? " (base_n cap of 2^24 hit)" : ""} after ` +
            `${doublings} doublings; returning ${design.samples.length / D} unique rows ` +
            "instead (duplicates are valid Saltelli samples)",
        );
        break;
      }
      baseN *= 2;
      doublings += 1;
      design = build();
      nExpanded = design.nExpanded;
    }
  }

  const nRuns = design.samples.length / D;
  if (verbose) {
    const duplicatesRemoved = nExpanded - nRuns;
    const duplicateFraction = nExpanded > 0 ? duplicatesRemoved / nExpanded : 0;
    console.info(
      `jaxgsa.sobol.sample: D=${D}, mode=${calcSecondOrder ? "second-order" : "first/total-order"}, ` +
        `base_n=${baseN}, n_runs=${nRuns}, n_expanded=${nExpanded}, ` +
        `duplicates_removed=${duplicatesRemoved} (${(duplicateFraction * 100).toFixed(1)}%), ` +
        `scramble=${scramble}`,
    );
  }

  return {
    samples: design.samples,
    nExpanded,
    expandedToUnique: design.expandedToUnique,
    baseN,
    nParams: D,
    calcSecondOrder,
  };
}