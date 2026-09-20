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

  const A_c = np.subtract(A, pooledMean.ref); // A consumed
  const B_c = np.subtract(B, pooledMean); // B + pooledMean consumed

  const pooledVar = np.divide(
    np.add(np.sum(np.square(A_c), 0), np.sum(np.square(B_c), 0)), // A_c, B_c consumed
    2 * N,
  );

  // pooledVar consumed on its last (raw) use
  return np.where(np.equal(pooledVar.ref, 0), np.nan, np.divide(1, pooledVar));
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

  // SAFETY: the caller supplies the scalar output as a host float64 buffer.
  let expanded = np.array(y as Float64Array<ArrayBuffer>, { dtype: np.float64 }); // (n_expanded,)

  if (options.expandedToUnique !== undefined) {
    // SAFETY: the sampler's expandedToUnique map is an Int32Array of row indices.
    const idx = np.array(options.expandedToUnique as Int32Array<ArrayBuffer>, {
      dtype: np.int32,
    });

    expanded = np.take(expanded, idx, 0);
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
  const safeScale = np.where(np.equal(yStd.ref, 0), 1, yStd); // yStd consumed
  const Ystd = np.divide(np.subtract(expanded, yMean), safeScale); // expanded, yMean, safeScale consumed

  // _separate_output_values: reshape (base_n, step) then slice.
  const grouped = np.reshape(Ystd, [baseN, step]); // Ystd consumed
  const A = grouped.ref.slice([], 0); // (N,)
  const B = grouped.ref.slice([], -1); // (N,)
  const AB = grouped.slice([], [1, D + 1]); // (N, D)  (grouped consumed)

  // _saltelli_jansen: _mauntz_kucherenko S1 + _jansen ST. The Python
  // composition computes the identical pooled inv_var twice; one call is
  // bit-identical, and it avoids leaking A and B (which the double .ref
  // call would leave at refcount 1).
  // Compute the squared-difference means from reductions and matrix-vector
  // products. This keeps the estimator from materialising one temporary
  // (N, D) product per first/total-order numerator.
  const sumA2 = np.sum(np.square(A.ref), 0);

  const bTimesAB = np.reshape(
    np.matmul(np.reshape(B.ref, [1, baseN]), AB.ref),
    [D],
  );

  const bTimesA = np.reshape(
    np.matmul(np.reshape(B.ref, [1, baseN]), np.reshape(A.ref, [baseN, 1])),
    [1],
  );

  const aTimesAB = np.reshape(
    np.matmul(np.reshape(A.ref, [1, baseN]), AB.ref),
    [D],
  );

  const sumAB2 = np.sum(np.square(AB), 0);

  const meanAMinusAB2 = np.divide(
    np.subtract(np.add(sumA2, sumAB2), np.multiply(2, aTimesAB)),
    baseN,
  );

  const invVar = pooledInvVar(A, B); // consumes A, B

  // Saltelli-Jansen uses the optimized Mauntz-Kucherenko first-order
  // cross-moment, while Jansen's squared-difference form supplies ST.
  const S1 = np.multiply(
    np.divide(np.subtract(bTimesAB, bTimesA), baseN),
    invVar.ref,
  );

  const ST = np.multiply(np.multiply(0.5, meanAMinusAB2), invVar);

  return {
    // SAFETY: jax-js dataSync returns float64 host buffers for the float64 estimator outputs.
    S1: S1.dataSync() as Float64Array,
    // SAFETY: jax-js dataSync returns float64 host buffers for the float64 estimator outputs.
    ST: ST.dataSync() as Float64Array,
  };
}
