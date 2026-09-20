import { numpy as np } from "@jax-js/jax";

/** Mirrors `_estimate` in src/jaxgsa/kucherenko/_analyze.py, run end-to-end in
 * jax-js `np` on the wasm device. Every input array is consumed (moved in);
 * the returned arrays own fresh memory.
 */
export function estimateKucherenko(
  fJoint: np.Array,
  fFirst: np.Array,
  fTotal: np.Array,
) {
  const variance = np.var_(fJoint.ref, 0); // (S,)

  const safeVariance = np.where(
    np.greater(variance.ref, 0),
    variance.ref,
    np.nan,
  );

  // Shift both S1 factors by one shared constant (the joint-block mean)
  // before the product. The estimator is algebraically unchanged: for any
  // shift c, mean((a-c)(b-c)) - mean(a-c) mean(b-c) == mean(ab) - mean(a)
  // mean(b). Numerically it is decisive: the uncentered product form loses
  // the O(1) covariance in rounding when Y carries a large mean, because
  // mean(f_joint f_first) and f0_joint f0_first are both O(mean^2). The ST
  // estimator is a squared difference and the variance is a central moment,
  // so both are already shift-safe.
  const f0Joint = np.mean(fJoint.ref, 0); // (S,)
  const gJoint = np.subtract(fJoint.ref, f0Joint.ref); // (N, S), centered
  // (D, N, S), same shift — f0Joint consumed on this, its last use
  const gFirst = np.subtract(fFirst, np.expandDims(f0Joint, [0, 1]));

  // The cross-moment is a per-(D, S) dot over N. Einsum avoids materialising
  // the (D, N, S) broadcast product that the literal multiply would create.
  const cross = np.einsum("dns,ns->ds", gFirst.ref, gJoint.ref);
  const crossMean = np.divide(cross, gJoint.shape[0]);
  const meanJoint = np.mean(gJoint, 0);
  const meanFirst = np.mean(gFirst, 1);

  const S1 = np.divide(
    np.subtract(crossMean, np.multiply(np.expandDims(meanJoint, 0), meanFirst)),
    safeVariance.ref,
  );

  const ST = np.divide(
    np.multiply(
      0.5,
      np.mean(np.square(np.subtract(np.expandDims(fJoint, 0), fTotal)), 1),
    ),
    safeVariance,
  );

  return { S1, ST, variance };
}

/** Mirrors `analyze` (block reshaping + estimator) for the 1-D output case.
 * `y` has shape (n,), which is promoted to (n, T=1, K=1) and reshaped
 * block-major to (2D+1, N, 1) — matching the Python (n_runs, T, K) promotion
 * with T=K=1. Both `x` and `y` are consumed.
 */
export function analyzeKucherenko(
  x: np.Array,
  y: np.Array,
) {
  const D = x.shape[1];
  const n = x.shape[0];
  const blocks = 2 * D + 1;
  const N = n / blocks;

  if (!Number.isInteger(N)) {
    throw new Error(
      `n_runs ${n} must be divisible by 2*D+1 = ${blocks} (got N = ${N})`,
    );
  }

  x.dispose();

  const F = np.reshape(y, [blocks, N, 1]); // (2D+1, N, 1) — block-major
  const fJoint = F.ref.slice(0, [], []); // (N, 1)
  const fFirst = F.ref.slice([1, D + 1], [], []); // (D, N, 1) — z redrawn given x_i
  const fTotal = F.slice([D + 1, blocks], [], []); // (D, N, 1) — x_i redrawn given z

  const { S1, ST, variance } = estimateKucherenko(fJoint, fFirst, fTotal);

  // SAFETY: jax-js dataSync returns float64 host buffers after the explicit reshapes.
  const s1 = np.reshape(S1, [D]).dataSync() as Float64Array;
  // SAFETY: jax-js dataSync returns float64 host buffers after the explicit reshapes.
  const st = np.reshape(ST, [D]).dataSync() as Float64Array;
  // SAFETY: jax-js dataSync returns float64 host buffers after the explicit reshape.
  const varOut = np.reshape(variance, [1]).dataSync() as Float64Array;

  return { S1: s1, ST: st, variance: varOut };
}
