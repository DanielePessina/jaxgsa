/**
 * Shapley-effect aggregation for the PCE backend (Owen, 2014): each expansion
 * term's variance share is split equally among the parameters active in it.
 *
 * Port of `src/jaxgsa/shapley/_engine.py` (`shapley_from_variances` +
 * `_normalize_partial_variances`) and the PCE branch of
 * `src/jaxgsa/shapley/_analyze.py`:
 *
 *   partial = coeffs[1:]**2
 *   membership = mi[1:] > 0                      (n_terms-1, D) 0/1
 *   V = partial / sum(partial)                   normalized variance shares
 *   card = sum(membership, axis=1)               |u| per term
 *   Sh = (V / card) @ membership                 split evenly per participant
 *   S1 = V @ (membership & (card == 1))          singletons only
 *   ST = V @ membership                          full credit
 *
 * The aggregation matmuls run in jax-js `np` (float64); the membership masks
 * are built host-side from the multi-index. `analyzeShapleyPce` reuses the
 * exact same fit as `analyzePce` (shared `fitPce`), so its S1/ST are the
 * same quantities up to normalization rounding.
 */

import { numpy as np } from "@jax-js/jax";
import type { ProblemSpec } from "../sampling";
import { fitPce, type PceFitOptions } from "../pce/analyze";
import { multiIndexColumns, sobolFromCoefficients, type MultiIndex } from "../pce/engine";

export interface ShapleyIndices {
  Sh: Float64Array;
  S1: Float64Array;
  ST: Float64Array;
}

/**
 * Shapley aggregation from PCE coefficients, mirroring the shared
 * `shapley_from_variances` tail. `coeffs` is the fitted expansion (host
 * Float64Array or a device np.Array from `fitPce`); `mi` is the multi-index
 * either as a `MultiIndex` or as its D columns (each length n_terms); `Sh`
 * sums to 1 per slice. When `coeffs` is an np.Array it is consumed (moved
 * in); pass `.ref` when the caller still needs it.
 */
export function shapleyFromPceCoefficients(
  coeffs: Float64Array | np.Array,
  mi: MultiIndex | Int32Array[],
  nParams: number,
): ShapleyIndices {
  const D = nParams;
  const miCols = Array.isArray(mi) ? mi : multiIndexColumns(mi);
  const nTerms = miCols[0].length;

  // SAFETY: coeffs is produced by the PCE fit as a host Float64Array.
  const coeffsNp =
    coeffs instanceof np.Array
      ? coeffs
      : np.array(coeffs as Float64Array<ArrayBuffer>, { dtype: np.float64 });

  // partial = coeffs[1:]**2; V = partial / sum(partial)
  const nonConst = coeffsNp.slice([1]); // (n_terms-1,) — coeffs consumed
  const partial = np.square(nonConst); // (n_terms-1,) — nonConst consumed
  const V = np.divide(partial, np.sum(partial.ref, 0)); // partial consumed

  // membership = mi[1:] > 0 as 0/1 float64; card = |u| per term.
  const membership = new Float64Array((nTerms - 1) * D);
  const card = new Float64Array(nTerms - 1);

  for (let t = 1; t < nTerms; t++) {
    let c = 0;

    for (let d = 0; d < D; d++) {
      if (miCols[d][t] > 0) {
        membership[(t - 1) * D + d] = 1;
        c++;
      }
    }

    card[t - 1] = c;
  }

  // membership & (card == 1): singleton rows carry exactly one active entry.
  const singletons = new Float64Array((nTerms - 1) * D);

  for (let t = 1; t < nTerms; t++) {
    if (card[t - 1] === 1) {
      for (let d = 0; d < D; d++) singletons[(t - 1) * D + d] = membership[(t - 1) * D + d];
    }
  }

  // SAFETY: membership is a host Float64Array built from the validated multi-index.
  const membershipNp = np
    .array(membership as Float64Array<ArrayBuffer>, { dtype: np.float64 })
    .reshape([nTerms - 1, D]); // (n_terms-1, D)

  // SAFETY: singletons is a host Float64Array built from the validated multi-index.
  const singletonsNp = np
    .array(singletons as Float64Array<ArrayBuffer>, { dtype: np.float64 })
    .reshape([nTerms - 1, D]); // (n_terms-1, D)

  const cardNp = np.sum(membershipNp.ref, 1); // (n_terms-1,) — membershipNp ref'd (used twice more)
  const VoverCard = np.divide(V.ref, cardNp); // (n_terms-1,) — cardNp consumed; V ref'd (used twice more)
  const Sh = np.matmul(VoverCard, membershipNp.ref); // (D,) — VoverCard consumed
  const S1 = np.matmul(V.ref, singletonsNp); // (D,) — singletonsNp consumed
  const ST = np.matmul(V, membershipNp); // (D,) — V, membershipNp consumed

  return {
    // SAFETY: jax-js dataSync returns float64 host buffers for float64 estimator outputs.
    Sh: Sh.dataSync() as Float64Array,
    // SAFETY: jax-js dataSync returns float64 host buffers for float64 estimator outputs.
    S1: S1.dataSync() as Float64Array,
    // SAFETY: jax-js dataSync returns float64 host buffers for float64 estimator outputs.
    ST: ST.dataSync() as Float64Array,
  };
}

/**
 * `analyzeShapleyPce`: run the same PCE fit as `analyzePce` (shared
 * `fitPce`) and aggregate the coefficients into Shapley effects plus the
 * matching S1/ST bounds. `x` is (N, D) row-major.
 */
export function analyzeShapleyPce(
  problem: ProblemSpec,
  x: Float64Array | number[],
  y: Float64Array | number[],
  options: PceFitOptions = {},
): ShapleyIndices {
  const fit = fitPce(problem, x, y, options);
  const sobol = sobolFromCoefficients(fit.coeffs.ref, fit.mi); // coeffs ref'd; shapley consumes raw
  const shapley = shapleyFromPceCoefficients(fit.coeffs, fit.mi, fit.mi.D);

  return { Sh: shapley.Sh, S1: sobol.S1, ST: sobol.ST };
}
