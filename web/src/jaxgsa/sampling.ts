/**
 * Shared sampling helpers used by the Sobol' and Morris ports, moved here
 * from `sobol/sample.ts` so both methods reuse the same marginal-transform
 * and deduplication machinery (the "design layer" of `jaxgsa._core.sampling`
 * and `jaxgsa.sampling`).
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
  /** Declared dependence structure -> the design is invalid. */
  correlation?: unknown;
}

// ---------------------------------------------------------------------------
// Marginal transforms (`_transform_samples`)
// ---------------------------------------------------------------------------

/** Unit-cube coordinates are clipped before the inverse-CDF (Python: UNIT_CLIP = 1e-12). */
export const UNIT_CLIP = 1e-12;

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
      "categorical marginals are not yet ported " +
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
