import { numpy as np } from "@jax-js/jax";
import type { MorrisDesign } from "./sample";

export interface MorrisMeasures {
  /** Mean elementary effect per parameter, shape (D,). */
  mu: Float64Array;
  /** Mean absolute elementary effect per parameter, shape (D,). */
  mu_star: Float64Array;
  /** Sample standard deviation (ddof=1) of the effects per parameter, shape (D,). */
  sigma: Float64Array;
}

/**
 * Port of `jaxgsa.morris.analyze` for scalar (1-D) output, run end-to-end in
 * jax-js `np` on the wasm device. `y` holds one output per unique design row
 * and is gathered into the expanded Morris layout via `expandedToUnique`
 * (the `expand_outputs` / `np.take(Y, expanded_to_unique, axis=0)` step),
 * then elementary effects are gathered by flattening the (r, D) index arrays
 * to 1-D, taking along axis 0, and reshaping back to (r, D):
 *
 *   Y_expanded = Y_unique[expandedToUnique]                    # (n_expanded,)
 *   ee = (Y_expanded[ee_idx_after] - Y_expanded[ee_idx_before]) / ee_delta  # (r, D)
 *   mu      = mean(ee, axis=0)
 *   mu_star = mean(|ee|, axis=0)
 *   sigma   = std(ee, axis=0, ddof=1)                          # / (r - 1)
 *
 * `ee_delta` broadcasts over the trailing output dims; with scalar output the
 * divide is elementwise. Every input array is consumed (moved in); the
 * returned Float64Arrays own fresh host memory.
 */
export function analyzeMorris(
  design: MorrisDesign,
  y: Float64Array | number[],
): MorrisMeasures {
  const r = design.nTrajectories;
  const D = design.nParams;

  // _expand_outputs: Y_expanded = Y_unique[expanded_to_unique]  (axis 0)
  // SAFETY: the caller's y is a host float64 buffer, and the method only reads its unique rows.
  let expanded = np.array(y as Float64Array<ArrayBuffer>, { dtype: np.float64 }); // (n_unique,)

  // SAFETY: the sampler stores expandedToUnique as an Int32Array of valid row indices.
  const e2u = np.array(design.expandedToUnique as Int32Array<ArrayBuffer>, {
    dtype: np.int32,
  });

  expanded = np.take(expanded, e2u, 0); // (n_expanded,)

  // Elementary effects: flatten the (r, D) index arrays, gather along axis 0,
  // reshape to (r, D). delta broadcasts over trailing output dims.
  // SAFETY: elementary-effect indices are emitted as Int32Array with shape (r, D).
  const afterFlat = np.array(design.eeIdxAfter as Int32Array<ArrayBuffer>, {
    dtype: np.int32,
  }); // (r * D,)

  // SAFETY: elementary-effect indices are emitted as Int32Array with shape (r, D).
  const beforeFlat = np.array(design.eeIdxBefore as Int32Array<ArrayBuffer>, {
    dtype: np.int32,
  });

  // SAFETY: elementary-effect deltas are emitted as Float64Array with shape (r, D).
  const deltaFlat = np.array(design.eeDelta as Float64Array<ArrayBuffer>, {
    dtype: np.float64,
  });

  const Yafter = np.take(expanded.ref, afterFlat, 0); // (r * D,)
  const Ybefore = np.take(expanded, beforeFlat, 0); // (r * D,)  (expanded consumed)
  const diff = np.subtract(Yafter, Ybefore); // (r * D,)  (Yafter, Ybefore consumed)
  const eeFlat = np.divide(diff, deltaFlat); // (r * D,)  (diff, deltaFlat consumed)
  const ee = np.reshape(eeFlat, [r, D]); // (r, D)  (eeFlat consumed)

  // _stats_from_ee
  const mu = np.mean(ee.ref, 0); // (D,)
  const muStar = np.mean(np.abs(ee.ref), 0); // (D,)
  // sigma = std(ee, axis=0, ddof=1) = sqrt(sum((ee - mean)^2, axis=0) / (r - 1))
  const centered = np.subtract(ee, np.expandDims(mu.ref, 0)); // (r, D)  (ee consumed)
  const sumSq = np.sum(np.square(centered), 0); // (D,)  (centered consumed)
  const sigma = np.sqrt(np.divide(sumSq, r - 1)); // (D,)  (sumSq consumed)

  return {
    // SAFETY: jax-js dataSync returns float64 host buffers for these float64 arrays.
    mu: mu.dataSync() as Float64Array,
    // SAFETY: jax-js dataSync returns float64 host buffers for these float64 arrays.
    mu_star: muStar.dataSync() as Float64Array,
    // SAFETY: jax-js dataSync returns float64 host buffers for these float64 arrays.
    sigma: sigma.dataSync() as Float64Array,
  };
}
