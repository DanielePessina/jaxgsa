/**
 * eFAST analysis: Fourier-spectrum first/total-order indices.
 *
 * Port of `src/jaxgsa/efast/_analyze.py` (`_compute_indices` +
 * `_indices_from_3d`, scalar path). Each parameter's search curve is an
 * ordered output sweep read by a discrete Fourier transform; the power at the
 * focal frequency (and its M harmonics) is the first-order contribution, and
 * the complementary low-frequency band gives the total contribution.
 *
 * The Python full-complex `jnp.fft.fft` is replaced by jax-js `numpy.fft.rfft`
 * (real input), whose positive-frequency bins align 1:1 with the hand-built
 * one-sided spectrum.
 */

import { defaultDevice, init, numpy as np } from "@jax-js/jax";
import type { EfastDesign } from "./sample";
import { frequencyPlan } from "./sample";

export interface EfastIndices {
  S1: Float64Array;
  ST: Float64Array;
}

interface CurveIndices {
  S1: number;
  ST: number;
}

let fftReady: Promise<void> | null = null;

function ensureFft(): Promise<void> {
  if (fftReady === null) {
    fftReady = (async () => {
      await init();
      defaultDevice("wasm");
    })();
  }

  return fftReady;
}

/** Per-curve S1/ST from the output sweep (`_compute_indices`). */
function computeIndices(
  curve: Float64Array,
  N: number,
  M: number,
  omega0: number,
  analysisMax: number,
): CurveIndices {
  // One-sided power spectrum: P[k] = (|rfft[k]| / N)^2, bin index == frequency.
  // SAFETY: curve is the caller's validated host Float64Array output slice.
  const spec = np.fft.rfft(
    np.array(curve as Float64Array<ArrayBuffer>, { dtype: np.float64 }),
    0,
  );

  const mag = np.hypot(spec.real, spec.imag);

  // SAFETY: jax-js dataSync returns a host Float64Array for the float64 magnitude.
  const F = mag.dataSync() as Float64Array;

  // V = 2*sum_{k=1}^{floor((N-1)/2)} P[k] + (N even ? P[N/2] : 0).
  let V = 0;

  for (let k = 1; k < N / 2; k++) V += F[k] * F[k];
  V *= 2;

  if (N % 2 === 0) V += F[N / 2] * F[N / 2];

  // D1 = 2*sum_{p=1..M} P[p * omega_0].
  let D1 = 0;

  for (let p = 1; p <= M; p++) {
    const f = p * omega0;
    D1 += F[f] * F[f];
  }

  D1 *= 2;

  // Dt = 2*sum_{k=1..analysis_max} P[k].
  let Dt = 0;

  for (let k = 1; k <= analysisMax; k++) Dt += F[k] * F[k];
  Dt *= 2;

  const constant = minMaxEqual(curve);

  return {
    S1: constant ? Number.NaN : D1 / V,
    ST: constant ? Number.NaN : 1 - Dt / V,
  };
}

function minMaxEqual(curve: Float64Array): boolean {
  let mn = curve[0];
  let mx = curve[0];

  for (let i = 1; i < curve.length; i++) {
    if (curve[i] < mn) mn = curve[i];

    if (curve[i] > mx) mx = curve[i];
  }

  return mn === mx;
}

/**
 * eFAST S1/ST for a scalar output. `y` is (n_per_curve * D,) in design row
 * order. Mirrors `jaxgsa.efast.analyze` for the scalar case (no diagnostics).
 */
export async function analyzeEfast(
  design: EfastDesign,
  y: Float64Array | number[],
): Promise<EfastIndices> {
  await ensureFft();

  const yf = y instanceof Float64Array ? y : Float64Array.from(y);
  const D = design.nParams;
  const N = design.nPerCurve;
  const M = design.M;
  const nRuns = N * D;

  if (yf.length !== nRuns) {
    throw new Error(
      `efast: design has ${nRuns} runs (n_per_curve=${N} * D=${D}) but y has ${yf.length} values`,
    );
  }

  const { omega0, analysisMax } = frequencyPlan(D, N, M);
  const S1 = new Float64Array(D);
  const ST = new Float64Array(D);

  for (let i = 0; i < D; i++) {
    const curve = yf.subarray(i * N, (i + 1) * N);
    const { S1: s1, ST: st } = computeIndices(curve, N, M, omega0, analysisMax);
    S1[i] = s1;
    ST[i] = st;
  }

  return { S1, ST };
}