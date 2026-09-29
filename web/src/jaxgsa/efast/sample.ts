/**
 * eFAST sampler: sinusoidal search-curve design (Cukier 1973 / Saltelli).
 *
 * Port of `src/jaxgsa/efast/_sampling.py`. For each of the D parameters,
 * `n_per_curve` points are drawn along one search curve where the focal
 * parameter oscillates at the highest frequency omega_0 and every other
 * parameter oscillates at a lower complementary frequency. The unit-cube
 * points are then transformed into physical units via the problem marginals.
 *
 * Sampling is host NumPy in Python, so this port is plain host JS; a tiny
 * seeded PRNG supplies the per-curve phase shifts.
 */

import { splitmix32 } from "@/jaxgsa/sobol/sampler";
import { transformSamples, type ProblemSpec } from "@/jaxgsa/sampling";

export interface EfastDesign {
  /** Rows in physical units, (n_per_curve * D, D) row-major. */
  samples: Float64Array;
  nParams: number;
  nPerCurve: number;
  M: number;
}

/** `_min_n_per_curve` / `_check_n_per_curve`: 4*M^2*max(D-1,1)+1. */
export function minNPerCurve(D: number, M: number): number {
  return 4 * M * M * Math.max(D - 1, 1) + 1;
}

/**
 * `_frequency_plan`: omega_0 and the complementary frequencies shared by the
 * sampler and the analyzer (`_sampling.py:246-276`).
 */
export function frequencyPlan(D: number, nPerCurve: number, M: number) {
  const omega0 = Math.floor((nPerCurve - 1) / (2 * M));
  const assignedMax = Math.floor(omega0 / (2 * M));
  const analysisMax = Math.floor(omega0 / 2);

  const omegaCompl: number[] = [];

  if (D > 1) {
    // floor(linspace(1, assigned_max, D-1)): D-1 points over [1, assigned_max].
    const n = D - 1;

    for (let i = 0; i < n; i++) {
      omegaCompl.push(Math.floor(1 + (i / (n - 1)) * (assignedMax - 1)));
    }
  }

  return { omega0, assignedMax, analysisMax, omegaCompl };
}

/**
 * Generate the eFAST design. `n_per_curve` must be >= `min_n_per_curve(D, M)`
 * (for D=1 the constraint is strict `> 4*M^2`).
 */
export function sampleEfast(
  problem: ProblemSpec,
  nPerCurve: number,
  options: { M?: number; seed?: number } = {},
): EfastDesign {
  const M = options.M ?? 4;
  const seed = options.seed ?? 0;
  const D = problem.names.length;

  if (!Number.isInteger(M) || M < 1) {
    throw new Error(`jaxgsa.efast.sample: M must be a positive integer, got ${M}`);
  }

  if (!Number.isInteger(nPerCurve) || nPerCurve < 1) {
    throw new Error(
      `jaxgsa.efast.sample: n_per_curve must be a positive integer, got ${nPerCurve}`,
    );
  }

  const floor = minNPerCurve(D, M);

  if (D === 1 ? nPerCurve <= 4 * M * M : nPerCurve < floor) {
    throw new Error(
      `jaxgsa.efast.sample: n_per_curve=${nPerCurve} is below the minimum ` +
        `${D === 1 ? 4 * M * M : floor} for D=${D}, M=${M}`,
    );
  }

  const { omega0, omegaCompl } = frequencyPlan(D, nPerCurve, M);
  const N = nPerCurve;
  const unit = new Float64Array(N * D * D);

  // Per-curve phases from a seeded PRNG (Python: np.random.default_rng).
  const rng = splitmix32(seed >>> 0);

  for (let i = 0; i < D; i++) {
    const phi = 2 * Math.PI * rng();
    const base = i * N;

    for (let j = 0; j < D; j++) {
      const omega = j === i ? omega0 : omegaCompl[j < i ? j : j - 1];

      for (let r = 0; r < N; r++) {
        const s = (2 * Math.PI * r) / N;
        const v = 0.5 + Math.asin(Math.sin(omega * s + phi)) / Math.PI;
        unit[(base + r) * D + j] = v;
      }
    }
  }

  return {
    samples: transformSamples(problem, unit),
    nParams: D,
    nPerCurve: N,
    M,
  };
}