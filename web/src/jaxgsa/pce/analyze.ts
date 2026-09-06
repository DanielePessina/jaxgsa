/**
 * PCE analysis entry point: fit an orthonormal polynomial surrogate on an
 * arbitrary (X, Y) point cloud and read first/total-order Sobol indices off
 * the coefficients (Sudret 2008). Port of the single-pass fit path of
 * `src/jaxgsa/pce/_analyze.py` (`_fit_pce_core` + `indices` for a scalar
 * output): autoOrder -> mapToReference -> buildDesignMatrix ->
 * gram + ridge*I -> linalg.solve -> sobolFromCoefficients, all in float64 on
 * the jax-js wasm device. S2, LOO diagnostics and bootstrapping are not
 * ported yet.
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
}

export interface PceIndices {
  S1: Float64Array;
  ST: Float64Array;
}

export interface PceFit {
  /** Fitted coefficients, (n_terms,) on the wasm device. Consumed by the caller. */
  coeffs: np.Array;
  /** The multi-index the coefficients are ordered against. */
  mi: MultiIndex;
  /** Effective polynomial degree after _auto_order. */
  order: number;
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

  // gram = Phi.T @ Phi + ridge * eye(n_terms)
  const PhiT = Phi.ref.transpose(); // (n_terms, N) — Phi ref'd (used once, in gram)
  const gram = np.matmul(PhiT.ref, Phi); // (n_terms, n_terms) — Phi consumed; PhiT ref'd (used again for B)
  const ridgeEye = np.multiply(
    ridge,
    np.eye(nTerms, nTerms, { dtype: np.float64 }),
  );
  const gramReg = np.add(gram, ridgeEye); // gram, ridgeEye consumed

  // coeffs = solve(gram, Phi.T @ Y)
  const Ynp = np.array(yf as Float64Array<ArrayBuffer>, { dtype: np.float64 }); // (N,)
  const B = np.matmul(PhiT, Ynp); // (n_terms,) — PhiT, Ynp consumed
  const coeffs = np.linalg.solve(gramReg, B); // (n_terms,) — gramReg, B consumed

  return { coeffs, mi, order: effOrder };
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
  return sobolFromCoefficients(fit.coeffs, fit.mi); // coeffs consumed
}