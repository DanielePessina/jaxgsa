/**
 * PCE analysis entry point: fit an orthonormal polynomial surrogate on an
 * arbitrary (X, Y) point cloud and read first/total-order Sobol indices off
 * the coefficients (Sudret 2008). Port of the single-pass fit path of
 * `src/jaxgsa/pce/_analyze.py` (`_fit_pce_core` + `indices` for a scalar
 * output): autoOrder -> mapToReference -> buildDesignMatrix ->
 * gram + ridge*I -> linalg.solve -> sobolFromCoefficients, all in float64 on
 * the jax-js wasm device. The fit-quality diagnostics (`explained_variance`
 * and the exact leave-one-out `cv_rmse`) are ported too. S2 and
 * bootstrapping are not ported yet.
 */

import { numpy as np } from "@jax-js/jax";
import type { ProblemSpec } from "../sampling";
import {
  autoOrder,
  buildDesignMatrix,
  buildMultiIndex,
  mapToReference,
  sobolFromCoefficients,
  type MultiIndex,
} from "./engine";

export interface PceFitOptions {
  /** Maximum total polynomial degree (default 3, as in the Python analyze). */
  order?: number;
  /** Tikhonov regularization added to the Gram diagonal (default 1e-8). */
  ridge?: number;
  /** Maximum ratio of terms to samples before order is reduced (default 0.5). */
  fitRatio?: number;
  /** Also compute `explained_variance` and `cv_rmse` (default false). */
  diagnostics?: boolean;
}

/**
 * Fit-quality numbers for one output slice, as `analyze` reports them.
 * `explainedVariance` is NaN for a constant output (no variance to explain).
 */
export interface PceDiagnostics {
  /** Sample variance of the fitted values over the sample variance of Y. */
  explainedVariance: number;
  /** Exact leave-one-out RMSE from the hat-matrix diagonal, in the units of Y. */
  cvRmse: number;
  /** Sample standard deviation of Y (the scale `cvRmse` is judged against). */
  stdY: number;
}

export interface PceIndices {
  S1: Float64Array;
  ST: Float64Array;
  /** Polynomial degree actually fitted (autoOrder may lower the request). */
  order: number;
  /** Present when `options.diagnostics` is true. */
  diagnostics?: PceDiagnostics;
}

export interface PceFit {
  /** Fitted coefficients, (n_terms,) on the wasm device. Consumed by the caller. */
  coeffs: np.Array;
  /** The multi-index the coefficients are ordered against. */
  mi: MultiIndex;
  /** Effective polynomial degree after _auto_order. */
  order: number;
  /** Present when `options.diagnostics` is true. */
  diagnostics?: PceDiagnostics;
}

/**
 * Leverage clip: keeps `1 - h_i` away from zero for a row the fit
 * interpolates exactly (`_LEVERAGE_CLIP` in `pce/_engine.py`).
 */
const LEVERAGE_CLIP = 1e-10;

/**
 * Port of the `explained_variance` and `cv_rmse` definitions in
 * `pce/_analyze.py`, on host arrays.
 *
 * - explained_variance = var(Phi c) / var(Y), both sample (ddof 0) variances
 *   over the fitted rows; NaN when Y is constant (`_is_constant_slice`).
 * - cv_rmse = sqrt(mean(((Y - Phi c) / (1 - h))^2)), h the hat-matrix
 *   diagonal clipped to [0, 1 - 1e-10] (`loo_error` + `hat_diagonal`).
 */
export function fitDiagnostics(
  y: Float64Array,
  yhat: Float64Array,
  leverage: Float64Array,
): PceDiagnostics {
  const n = y.length;
  let mean = 0;
  let meanHat = 0;
  let yMin = Infinity;
  let yMax = -Infinity;

  for (let i = 0; i < n; i++) {
    mean += y[i];
    meanHat += yhat[i];
    yMin = Math.min(yMin, y[i]);
    yMax = Math.max(yMax, y[i]);
  }

  mean /= n;
  meanHat /= n;

  let varY = 0;
  let varHat = 0;
  let sse = 0;

  for (let i = 0; i < n; i++) {
    varY += (y[i] - mean) ** 2;
    varHat += (yhat[i] - meanHat) ** 2;
    const h = Math.min(Math.max(leverage[i], 0), 1 - LEVERAGE_CLIP);
    sse += ((y[i] - yhat[i]) / (1 - h)) ** 2;
  }

  varY /= n;
  varHat /= n;

  return {
    explainedVariance: yMax === yMin ? NaN : varHat / varY,
    cvRmse: Math.sqrt(sse / n),
    stdY: Math.sqrt(varY),
  };
}

/**
 * The shared PCE fit, mirroring `_fit_pce_core` for a scalar output:
 *
 *   gram = Phi^T Phi + ridge * eye(n_terms)
 *   coeffs = solve(gram, Phi^T @ Y)
 *
 * `x` is (N, D) row-major. Returns the coefficients on device; the caller
 * owns the returned np.Array and must consume it exactly once (`.ref` if it
 * needs it twice).
 */
export function fitPce(
  problem: ProblemSpec,
  x: Float64Array | number[],
  y: Float64Array | number[],
  options: PceFitOptions = {},
): PceFit {
  const order = options.order ?? 3;
  const ridge = options.ridge ?? 1e-8;
  const fitRatio = options.fitRatio ?? 0.5;

  const xf = x instanceof Float64Array ? x : Float64Array.from(x);
  const yf = y instanceof Float64Array ? y : Float64Array.from(y);
  const D = problem.names.length;
  const N = xf.length / D;

  if (!Number.isInteger(N)) {
    throw new Error(`pce: x length ${xf.length} is not a multiple of D=${D}`);
  }

  const effOrder = autoOrder(D, N, order, fitRatio);
  const mi = buildMultiIndex(D, effOrder);
  const nTerms = mi.nTerms;

  const { xRef, inputTypes } = mapToReference(xf, problem);
  const Phi = buildDesignMatrix(xRef, mi, inputTypes, effOrder); // (N, n_terms)
  const PhiKeep = options.diagnostics ? Phi.ref : null; // kept for the diagnostics below

  // gram = Phi.T @ Phi + ridge * eye(n_terms)
  const PhiT = Phi.ref.transpose(); // (n_terms, N) — Phi ref'd (used once, in gram)
  const gram = np.matmul(PhiT.ref, Phi); // (n_terms, n_terms) — Phi consumed; PhiT ref'd (used again for B)

  const ridgeEye = np.multiply(
    ridge,
    np.eye(nTerms, nTerms, { dtype: np.float64 }),
  );

  const gramReg = np.add(gram, ridgeEye); // gram, ridgeEye consumed

  // coeffs = solve(gram, Phi.T @ Y)
  // SAFETY: yf is the validated scalar output copied into a host Float64Array.
  const Ynp = np.array(yf as Float64Array<ArrayBuffer>, { dtype: np.float64 }); // (N,)
  const B = np.matmul(options.diagnostics ? PhiT.ref : PhiT, Ynp); // (n_terms,) — Ynp consumed
  const coeffs = np.linalg.solve(options.diagnostics ? gramReg.ref : gramReg, B); // B consumed

  if (!PhiKeep) return { coeffs, mi, order: effOrder };

  // yhat = Phi c, and the leverage h_i = phi_i (Phi^T Phi + ridge I)^-1 phi_i^T
  // read off Z = solve(gram, Phi^T) as the column-wise dot of Phi^T and Z.
  const yhat = np.matmul(PhiKeep.ref, coeffs.ref).dataSync();
  const Z = np.linalg.solve(gramReg, PhiT); // (n_terms, N) — gramReg, PhiT consumed
  const leverage = np.sum(np.multiply(Z, PhiKeep.transpose()), 0).dataSync(); // PhiKeep consumed

  return {
    coeffs,
    mi,
    order: effOrder,
    // SAFETY: jax-js dataSync returns float64 host buffers for float64 arrays.
    diagnostics: fitDiagnostics(yf, yhat as Float64Array, leverage as Float64Array),
  };
}

/**
 * `analyzePce`: fit the expansion on the given cloud and return first/total
 * Sobol indices, mirroring `jaxgsa.pce.analyze` for scalar output (no
 * diagnostics). `x` is (N, D) row-major.
 */
export function analyzePce(
  problem: ProblemSpec,
  x: Float64Array | number[],
  y: Float64Array | number[],
  options: PceFitOptions = {},
): PceIndices {
  const fit = fitPce(problem, x, y, options);

  const { S1, ST } = sobolFromCoefficients(fit.coeffs, fit.mi); // coeffs consumed

  return { S1, ST, order: fit.order, diagnostics: fit.diagnostics };
}
