/**
 * PAWN index port (scalar output): the given-data Kolmogorov-Smirnov-based
 * distribution sensitivity measure of Pianosi & Wagener.
 *
 * Port of `src/jaxgsa/pawn/_analyze.py` (`_pawn_core` + `indices`) restricted
 * to the scalar-output, continuous-marginals path with `n_bootstrap = 0` (the
 * default): CDF-transform X to the unit interval, bin into equal-width
 * (equal-probability) bins, then per parameter compute the KS distance
 * between the unconditional and each bin-conditional output CDF, aggregated
 * by the median (default) into one index per parameter.
 *
 * The Python `searchsorted` binning is replaced by the exact equal-width
 * formula `idx = clip(floor(X_u01 * n_bins), 0, n_bins-1)`; the median
 * aggregation and the KS kernel run as host JS loops.
 */

import type { MarginalSpec, ProblemSpec } from "@/jaxgsa/sampling";
import { normalCdf } from "@/jaxgsa/sampling";

const UNIT_CLIP = 1e-12;

export interface PawnOptions {
  /** Number of equal-width bins (default 10). */
  nBins?: number;
  /** Aggregation over bins: "median" (default), "max", or "mean". */
  statistic?: "median" | "max" | "mean";
}

export interface PawnIndices {
  /** PAWN index per parameter, (D,). */
  pawn: Float64Array;
}

/** CDF-transform one column to the unit interval (`cdf_to_unit_interval`). */
function cdfToUnitInterval(col: Float64Array, spec: MarginalSpec): Float64Array {
  const n = col.length;
  const out = new Float64Array(n);

  if (spec.kind === "uniform") {
    const inv = 1 / (spec.high - spec.low);

    for (let i = 0; i < n; i++) out[i] = (col[i] - spec.low) * inv;

    return out;
  }

  if (spec.kind === "categorical") {
    throw new Error("pawn: categorical marginals are not yet ported");
  }

  // gaussian (truncated or not): Phi((x - mu) / sigma), clipped to (0, 1).
  const std = Math.sqrt(spec.variance);
  const a = spec.low === undefined ? -Infinity : (spec.low - spec.mean) / std;
  const b = spec.high === undefined ? Infinity : (spec.high - spec.mean) / std;
  const pa = a === -Infinity ? 0 : normalCdf(a);
  const pb = b === Infinity ? 1 : normalCdf(b);

  for (let i = 0; i < n; i++) {
    const z = (col[i] - spec.mean) / std;
    let u = normalCdf(z);

    if (spec.low !== undefined || spec.high !== undefined) {
      u = (u - pa) / (pb - pa);
    }

    // Clip in-range Gaussian CDF values off the exact 0/1 tails (Python
    // clips to [1e-12, 1-1e-12] only where the value is already in range).
    out[i] = u > UNIT_CLIP && u < 1 - UNIT_CLIP
      ? Math.min(Math.max(u, UNIT_CLIP), 1 - UNIT_CLIP)
      : u;
  }

  return out;
}

/** The KS kernel for one output column, returning (D, n_bins) KS values. */
function pawnKs(
  y: Float64Array,
  binIdx: Int32Array,
  D: number,
  nBins: number,
): Float64Array {
  const N = y.length;

  // Sort the outputs; gate at tie-group ends.
  const order = Array.from({ length: N }, (_, i) => i);
  order.sort((a, b) => y[a] - y[b]);

  const ySorted = new Float64Array(N);
  const binSorted = new Int32Array(N * D);
  const isGroupEnd = new Uint8Array(N);

  for (let i = 0; i < N; i++) {
    ySorted[i] = y[order[i]];

    for (let d = 0; d < D; d++) binSorted[i * D + d] = binIdx[order[i] * D + d];

    isGroupEnd[i] = i === N - 1 || y[order[i]] !== y[order[i + 1]] ? 1 : 0;
  }

  const ks = new Float64Array(D * nBins);

  for (let d = 0; d < D; d++) {
    const cnt = new Float64Array(nBins);

    for (let i = 0; i < N; i++) {
      const b = binSorted[i * D + d];

      if (b >= 0) cnt[b] += 1;
    }

    const cum = new Float64Array(nBins);

    for (let i = 0; i < N; i++) {
      const b = binSorted[i * D + d];

      if (b >= 0) cum[b] += 1;

      if (isGroupEnd[i] === 1) {
        const uncond = (i + 1) / N;

        for (let bin = 0; bin < nBins; bin++) {
          const ccdf = cnt[bin] > 0 ? cum[bin] / cnt[bin] : 0;
          const diff = Math.abs(uncond - ccdf);

          if (diff > ks[d * nBins + bin]) ks[d * nBins + bin] = diff;
        }
      }
    }

    // Bins with fewer than 2 members are NaN (dropped from aggregation).
    for (let bin = 0; bin < nBins; bin++) {
      if (cnt[bin] < 2) ks[d * nBins + bin] = Number.NaN;
    }
  }

  return ks;
}

/** Aggregate bin KS values per parameter (`_aggregate_ks`). */
function aggregate(ks: Float64Array, D: number, nBins: number, statistic: PawnOptions["statistic"]): Float64Array {
  const out = new Float64Array(D);

  for (let d = 0; d < D; d++) {
    const vals: number[] = [];

    for (let bin = 0; bin < nBins; bin++) {
      const v = ks[d * nBins + bin];

      if (!Number.isNaN(v)) vals.push(v);
    }

    if (statistic === "mean") {
      let s = 0;

      for (const v of vals) s += v;
      out[d] = vals.length > 0 ? s / vals.length : Number.NaN;
    } else if (statistic === "max") {
      let m = -Infinity;

      for (const v of vals) if (v > m) m = v;
      out[d] = vals.length > 0 ? m : Number.NaN;
    } else {
      // median: middle element, averaging the two middles when even (NumPy).
      vals.sort((a, b) => a - b);

      const mid = vals.length / 2;

      out[d] = vals.length > 0
        ? vals.length % 2 === 1
          ? vals[Math.floor(mid)]
          : (vals[mid - 1] + vals[mid]) / 2
        : Number.NaN;
    }
  }

  return out;
}

/**
 * PAWN index for a scalar output. `x` is (N, D) row-major; `y` is (N,).
 * Mirrors `jaxgsa.pawn.analyze` with the defaults (`n_bins=10`,
 * `statistic="median"`, `n_bootstrap=0`).
 */
export function analyzePawn(
  problem: ProblemSpec,
  x: Float64Array | number[],
  y: Float64Array | number[],
  options: PawnOptions = {},
): PawnIndices {
  const xf = x instanceof Float64Array ? x : Float64Array.from(x);
  const yf = y instanceof Float64Array ? y : Float64Array.from(y);
  const D = problem.names.length;
  const N = xf.length / D;
  const nBins = options.nBins ?? 10;
  const statistic = options.statistic ?? "median";

  if (!Number.isInteger(N) || N === 0) {
    throw new Error(`pawn: x length ${xf.length} is not a multiple of D=${D}`);
  }

  if (yf.length !== N) {
    throw new Error(`pawn: x has ${N} rows but y has ${yf.length} values`);
  }

  if (nBins < 2) {
    throw new Error(`pawn: n_bins must be >= 2, got ${nBins}`);
  }

  // Equal-width binning on the CDF-transformed unit interval.
  const binIdx = new Int32Array(N * D);

  for (let d = 0; d < D; d++) {
    const col = new Float64Array(N);

    for (let i = 0; i < N; i++) col[i] = xf[i * D + d];

    const u = cdfToUnitInterval(col, problem.marginals[d]);

    for (let i = 0; i < N; i++) {
      const inRange = u[i] >= 0 && u[i] <= 1;
      const idx = Math.floor(u[i] * nBins);
      const clipped = Math.min(Math.max(idx, 0), nBins - 1);

      binIdx[i * D + d] = inRange ? clipped : -1;
    }
  }

  const ks = pawnKs(yf, binIdx, D, nBins);
  const pawn = aggregate(ks, D, nBins, statistic);

  return { pawn };
}