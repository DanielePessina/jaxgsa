import { numpy as np } from "@jax-js/jax";

export interface AnalyzeSobolOptions {
  /** Second-order indices (S2) are not ported yet; must be false (default). */
  calcSecondOrder?: boolean;
  /**
   * Optional sanity check against the base_n implied by `y.length / (D + 2)`.
   * When omitted, base_n is derived from the expanded row count.
   */
  baseN?: number;
  /**
   * When given, `y` holds one output per unique design row and is gathered
   * into the expanded Saltelli layout via `expanded_to_unique[i]` (mirrors
   * `expand_outputs` / `np.take(Y, expanded_to_unique, axis=0)`).
   */
  expandedToUnique?: Int32Array;
}

export interface SobolIndices {
  S1: Float64Array;
  ST: Float64Array;
}

/**
 * `_pooled_inv_var` from src/jaxgsa/sobol/_estimators.py: `1 / Var(concat(A, B))`
 * with ddof=0, or NaN when the pooled variance is exactly zero. Centering
 * before squaring avoids catastrophic cancellation for large-magnitude outputs.
 * `A` and `B` are consumed.
 */
function pooledInvVar(A: np.Array, B: np.Array): np.Array {
  const N = A.shape[0];
  const pooledMean = np.divide(
    np.add(np.mean(A.ref, 0), np.mean(B.ref, 0)),
    2,
  );
  const A_c = np.subtract(A.ref, pooledMean.ref);
  const B_c = np.subtract(B.ref, pooledMean.ref);
  const pooledVar = np.divide(
    np.add(np.sum(np.square(A_c.ref), 0), np.sum(np.square(B_c.ref), 0)),
    2 * N,
  );
  return np.where(np.equal(pooledVar.ref, 0), np.nan, np.divide(1, pooledVar.ref));
}

/**
 * Port of `jaxgsa.sobol.analyze` for scalar (1-D) outputs with the default
 * `saltelli-jansen` estimator, first/total order only. The whole pipeline
 * runs in float64 in jax-js `np` on the wasm device, mirroring
 * `_standardize_outputs`, `_separate_output_values`, `_mauntz_kucherenko`
 * (S1) and `_jansen` (ST) exactly:
 *
 *   Y_std = (Y - mean(Y)) / std(Y)                 over the sample axis
 *   grouped = Y_std.reshape(base_n, D + 2)         step = D + 2
 *   A = grouped[:, 0];  B = grouped[:, -1];  AB = grouped[:, 1 : D + 1]
 *   S1 = mean(B[:, None] * (AB - A[:, None])) * inv_var
 *   ST = 0.5 * mean((A[:, None] - AB)**2) * inv_var
 *   inv_var = 1 / pooled var over concat(A, B), NaN when zero
 *
 * Every input array is consumed (moved in); the returned Float64Arrays own
 * fresh host memory.
 */
export function analyzeSobol(
  y: Float64Array | number[],
  nParams: number,
  options: AnalyzeSobolOptions = {},
): SobolIndices {
  const calcSecondOrder = options.calcSecondOrder ?? false;
  if (calcSecondOrder) {
    throw new Error(
      "jaxgsa.sobol.analyze: calcSecondOrder=true is not supported yet " +
        "(the S2 estimator needs the BA blocks and is not ported)",
    );
  }
  const D = nParams;
  const step = D + 2;

  let expanded = np.array(y as Float64Array<ArrayBuffer>, { dtype: np.float64 }); // (n_expanded,)
  if (options.expandedToUnique !== undefined) {
    const idx = np.array(options.expandedToUnique as Int32Array<ArrayBuffer>, {
      dtype: np.int32,
    });
    expanded = np.take(expanded.ref, idx, 0);
  }

  const nExpanded = expanded.shape[0];
  const baseN = nExpanded / step;
  if (!Number.isInteger(baseN)) {
    throw new Error(
      `jaxgsa.sobol.analyze: n_expanded=${nExpanded} is not a multiple of the ` +
        `Saltelli group size step=${step}; the design cannot be split into ` +
        `whole groups (base_n = n_expanded / step = ${nExpanded} / ${step})`,
    );
  }
  if (options.baseN !== undefined && options.baseN !== baseN) {
    throw new Error(
      `jaxgsa.sobol.analyze: baseN=${options.baseN} does not match the ` +
        `derived base_n=${baseN} (n_expanded / step)`,
    );
  }

  // _standardize_outputs: (Y - mean) / std over the sample axis (ddof=0),
  // with zero std replaced by 1 so constant slices stay all-zero (NaN-free).
  const yMean = np.mean(expanded.ref, 0);
  const yStd = np.std(expanded.ref, 0);
  const safeScale = np.where(np.equal(yStd.ref, 0), 1, yStd.ref);
  const Ystd = np.divide(np.subtract(expanded.ref, yMean.ref), safeScale.ref);

  // _separate_output_values: reshape (base_n, step) then slice.
  const grouped = np.reshape(Ystd.ref, [baseN, step]);
  const A = grouped.ref.slice([], 0); // (N,)
  const B = grouped.ref.slice([], -1); // (N,)
  const AB = grouped.ref.slice([], [1, D + 1]); // (N, D)

  // _saltelli_jansen: _mauntz_kucherenko S1 + _jansen ST, one pooled
  // inv_var each (as the Python composition computes it twice).
  const invVarS1 = pooledInvVar(A, B);
  const invVarST = pooledInvVar(A, B);

  // S1 = mean(B[:, None] * (AB - A[:, None]), axis=0) * inv_var
  const Acol = np.expandDims(A.ref, 1); // (N, 1)
  const Bcol = np.expandDims(B.ref, 1); // (N, 1)
  const diff = np.subtract(AB.ref, Acol.ref); // (N, D)
  const prod = np.multiply(Bcol.ref, diff.ref); // (N, D)
  const S1 = np.multiply(np.mean(prod.ref, 0), invVarS1.ref); // (D,)

  // ST = 0.5 * mean((A[:, None] - AB)**2, axis=0) * inv_var
  const diff2 = np.subtract(Acol.ref, AB.ref); // (N, D)
  const sq = np.square(diff2.ref); // (N, D)
  const meanSq = np.mean(sq.ref, 0); // (D,)
  const ST = np.multiply(np.multiply(0.5, meanSq.ref), invVarST.ref); // (D,)

  return {
    S1: S1.dataSync() as Float64Array,
    ST: ST.dataSync() as Float64Array,
  };
}