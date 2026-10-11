/**
 * Kucherenko conditional-copula design sampler, independent-problem case
 * (problem.correlation is None): with an identity copula the conditional
 * redraws collapse to fresh independent draws, so the latent-normal
 * construction cancels and the unit-cube design is exactly the Saltelli
 * column-swap layout — joint block, then one first-order block and one
 * total-order block per parameter, `base_n * (2D + 1)` rows total.
 *
 * Port of `jaxgsa.kucherenko.sample` restricted to the independent problem.
 * The layout is BLOCK-MAJOR, matching the Python concatenation of
 * `[Z_joint, first_0..first_{D-1}, total_0..total_{D-1}]` and the
 * `(2D+1, N, 1)` reshape `analyzeKucherenko` applies to `y`:
 *
 *   first-order block i: keep x_i from the joint row, redraw the rest
 *                        (python: assemble_latent(i, others, Z_joint[:, i],
 *                         redraw[:, others]))
 *   total block i:       keep the joint others, redraw x_i
 *                        (python: assemble_latent(i, others, redraw[:, i],
 *                         Z_joint[:, others]))
 */

import { sobolSequence } from "@/jaxgsa/sobol/sampler";
import {
  nextPowerOfTwo,
  transformSamples,
  validateProblem,
  type ProblemSpec,
} from "@/jaxgsa/sampling";

export interface KucherenkoDesign {
  /** Rows in physical units, (n_runs, D) row-major, block-major layout. */
  samples: Float64Array;
  nParams: number;
  baseN: number;
}

/** Design blocks per base point: `2D + 1` (singletons mode). */
export function kucherenkoBlocks(nParams: number): number {
  return 2 * nParams + 1;
}

/**
 * Base points per block for a minimum run budget.
 *
 * Mirrors `n = max(2, _next_power_of_2(-(-n_samples // n_blocks)))` in
 * `jaxgsa.kucherenko.sample` (src/jaxgsa/kucherenko/_sampling.py).
 */
export function kucherenkoBaseN(nSamples: number, nParams: number): number {
  return nextPowerOfTwo(Math.max(2, Math.ceil(nSamples / kucherenkoBlocks(nParams))));
}

/** Model runs in the design for a minimum run budget: `base_n * (2D + 1)`. */
export function kucherenkoRuns(nSamples: number, nParams: number): number {
  return kucherenkoBaseN(nSamples, nParams) * kucherenkoBlocks(nParams);
}

/** `nSamples` is the minimum number of model runs, as in `jaxgsa.kucherenko.sample`. */
export function sampleKucherenkoDesign(
  problem: ProblemSpec,
  nSamples: number,
  seed: number,
): KucherenkoDesign {
  validateProblem(problem, "kucherenko");

  if (!Number.isInteger(nSamples) || nSamples < 2) {
    throw new Error(
      `jaxgsa.kucherenko.sample: n_samples must be a whole number, at least 2, got ${nSamples}`,
    );
  }

  const D = problem.names.length;
  const baseN = kucherenkoBaseN(nSamples, D);
  const blocks = kucherenkoBlocks(D);
  const draws = sobolSequence(2 * D, baseN, true, seed);
  // Block-major layout: block b holds base points 0..baseN-1 in rows
  // [b*baseN, (b+1)*baseN).
  const unit = new Float64Array(baseN * blocks * D);

  for (let k = 0; k < baseN; k++) {
    const b = k * 2 * D;

    // Joint block: row k.
    for (let j = 0; j < D; j++) unit[k * D + j] = draws[b + j];

    // First-order blocks: keep x_i from the joint row, redraw the rest.
    for (let i = 0; i < D; i++) {
      const oo = ((1 + i) * baseN + k) * D;

      for (let j = 0; j < D; j++) unit[oo + j] = draws[b + D + j];
      unit[oo + i] = draws[b + i]; // x_i kept from joint
    }

    // Total blocks: keep the joint others, redraw x_i.
    for (let i = 0; i < D; i++) {
      const oo = ((1 + D + i) * baseN + k) * D;

      for (let j = 0; j < D; j++) unit[oo + j] = draws[b + j];
      unit[oo + i] = draws[b + D + i]; // x_i redrawn
    }
  }

  return { samples: transformSamples(problem, unit), nParams: D, baseN };
}