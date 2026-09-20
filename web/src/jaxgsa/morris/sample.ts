import { splitmix32 } from "../sobol/sampler";
import {
  dedupeDesign,
  validateProblem,
  type ProblemSpec,
} from "../sampling";

/**
 * Port of `jaxgsa.morris.sample` (trajectory scheme only) to plain TS.
 *
 * The trajectory design walks each of `r` independent paths of `D + 1`
 * points on a coarse `num_levels` grid, perturbing every parameter exactly
 * once per path (Morris 1991). Points are built on an integer half-level
 * grid, in units of `1 / (2 (p - 1))`, and converted to float once at the
 * end so equal grid points are bitwise identical across trajectories — which
 * is what lets exact deduplication collapse them. The unit-cube design is
 * deduplicated (first-occurrence, bitwise row equality) BEFORE the marginal
 * transform, and the elementary-effect bookkeeping (`ee_idx_*`, `ee_delta`)
 * stays in expanded-row space, which is unaffected by dedup.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface MorrisDesign {
  /** Unique rows in physical units, shape (nUnique, nParams), row-major. */
  samples: Float64Array;
  /** Map from each expanded row to its retained unique row, shape (nExpanded,). */
  expandedToUnique: Int32Array;
  /** Row count of the full expanded design before deduplication, r * (D + 1). */
  nExpanded: number;
  /** Number of trajectories r, the Morris repetition unit. */
  nTrajectories: number;
  /** Grid levels p used by the trajectory design. */
  numLevels: number;
  /** Expanded-row index of the perturbed point of each elementary effect, row-major (r, D). */
  eeIdxAfter: Int32Array;
  /** Expanded-row index of the reference point of each elementary effect, row-major (r, D). */
  eeIdxBefore: Int32Array;
  /** Signed unit-cube step of each elementary effect, row-major (r, D). */
  eeDelta: Float64Array;
  nParams: number;
}

export interface SampleMorrisOptions {
  /** Grid levels p (default 4, step delta = p / (2 (p - 1))). */
  numLevels?: number;
  /** Seed for the deterministic splitmix32 stream (default 0). */
  seed?: number;
  /** Print a summary like the Python sample() (default true). */
  verbose?: boolean;
  /** Design generator. Only "trajectory" is ported; "radial" raises. */
  method?: "trajectory" | "radial";
  /** Tail probability q pulled from each open side of every Gaussian marginal (default 1e-4). */
  truncationQuantile?: number;
}

// ---------------------------------------------------------------------------
// PRNG helpers on top of splitmix32
// ---------------------------------------------------------------------------

type Rng = () => number;

/** Uniform integer in [0, n), one splitmix32 draw modulo n. */
function randomInt(rng: Rng, n: number): number {
  return rng() % n;
}

/** Fisher-Yates shuffle of [0, n), mirroring `rng.permutation(D)`. */
function permutation(rng: Rng, n: number): Int32Array {
  const p = new Int32Array(n);

  for (let i = 0; i < n; i++) p[i] = i;

  for (let i = n - 1; i > 0; i--) {
    const j = randomInt(rng, i + 1);
    const tmp = p[i];
    p[i] = p[j];
    p[j] = tmp;
  }

  return p;
}

// ---------------------------------------------------------------------------
// Trajectory construction (`_build_trajectories`)
// ---------------------------------------------------------------------------

interface TrajectoryDesign {
  expandedUnit: Float64Array;
  eeIdxAfter: Int32Array;
  eeIdxBefore: Int32Array;
  eeDelta: Float64Array;
}

/**
 * Generate Morris trajectories on the `num_levels` grid, the verbatim port
 * of `_build_trajectories` in src/jaxgsa/morris/_sampling.py. Per-trajectory
 * randomness is consumed in the Python order: base levels, step signs, then
 * the parameter order permutation.
 */
function buildTrajectories(
  nTrajectories: number,
  nParams: number,
  numLevels: number,
  rng: Rng,
): TrajectoryDesign {
  const D = nParams;
  const p = numLevels;
  // delta = p / (2 (p - 1)) expressed in half-level units of 1 / (2 (p - 1)).
  const deltaInt = p;
  const delta = p / (2.0 * (p - 1));
  // A base level l / (p - 1) must leave room for a +delta step: l <= p/2 - 1.
  const nStartLevels = p >> 1;

  const nRows = nTrajectories * (D + 1);
  const levelsInt = new Int32Array(nRows * D);
  const eeIdxAfter = new Int32Array(nTrajectories * D);
  const eeIdxBefore = new Int32Array(nTrajectories * D);
  const eeDelta = new Float64Array(nTrajectories * D);

  const x = new Int32Array(D);
  const base = new Int32Array(D);
  const signs = new Int32Array(D);

  for (let j = 0; j < nTrajectories; j++) {
    // Per-trajectory draws, fixed order: base levels, step signs, order.
    for (let i = 0; i < D; i++) base[i] = 2 * randomInt(rng, nStartLevels); // even = on-grid

    for (let i = 0; i < D; i++) signs[i] = 2 * randomInt(rng, 2) - 1; // each +/-1
    const perm = permutation(rng, D);

    // A -delta step needs headroom below, so shift its start up by delta.
    for (let i = 0; i < D; i++) x[i] = base[i] + (signs[i] < 0 ? deltaInt : 0);
    const offset = j * (D + 1);

    for (let i = 0; i < D; i++) levelsInt[offset * D + i] = x[i];

    for (let s = 0; s < D; s++) {
      const i = perm[s];
      x[i] += signs[i] * deltaInt;

      for (let k = 0; k < D; k++) levelsInt[(offset + s + 1) * D + k] = x[k];
      eeIdxBefore[j * D + i] = offset + s;
      eeIdxAfter[j * D + i] = offset + s + 1;
      eeDelta[j * D + i] = signs[i] * delta;
    }
  }

  const denom = 2 * (p - 1);
  const expandedUnit = new Float64Array(levelsInt.length);

  for (let i = 0; i < levelsInt.length; i++) expandedUnit[i] = levelsInt[i] / denom;

  return { expandedUnit, eeIdxAfter, eeIdxBefore, eeDelta };
}

// ---------------------------------------------------------------------------
// Open-side squash (`_squash_open_sides`)
// ---------------------------------------------------------------------------

/**
 * Pull the design away from the unit-cube faces on open Gaussian sides.
 * A Morris design touches the unit-cube boundaries exactly, and an unbounded
 * inverse CDF maps 0 and 1 to -inf and +inf, so each genuinely open side is
 * pulled in by the tail probability `q` before the transform. Sides the
 * problem already bounds with `low`/`high` never move (a two-sided truncated
 * Gaussian gets no squash). The squash is affine per dimension, so the
 * elementary-effect step is rescaled by the same factor to keep the divisor
 * equal to the coordinate difference the design really takes.
 */
function squashOpenSides(
  expandedUnit: Float64Array,
  eeDelta: Float64Array,
  problem: ProblemSpec,
  D: number,
  q: number,
): void {
  for (let idx = 0; idx < D; idx++) {
    const spec = problem.marginals[idx];

    if (spec.kind !== "gaussian") continue;
    const loTarget = spec.low === undefined ? q : 0.0;
    const hiTarget = spec.high === undefined ? 1.0 - q : 1.0;
    const scale = hiTarget - loTarget;

    if (scale === 1.0) continue; // both sides already bounded, nothing to squash
    const nRows = expandedUnit.length / D;

    for (let r = 0; r < nRows; r++) {
      expandedUnit[r * D + idx] = loTarget + expandedUnit[r * D + idx] * scale;
    }

    const nEffects = eeDelta.length / D;

    for (let j = 0; j < nEffects; j++) eeDelta[j * D + idx] *= scale;
  }
}

// ---------------------------------------------------------------------------
// Public entry point (`sample`)
// ---------------------------------------------------------------------------

/**
 * Generate a trajectory Morris design for screening. Builds `nTrajectories`
 * paths of `D + 1` points each (r * (D + 1) expanded rows), deduplicates
 * exact duplicate rows (common in low dimensions on a coarse grid), and
 * returns only the unique rows the user must evaluate plus the elementary-
 * effect bookkeeping for `analyzeMorris`.
 */
export function sampleMorris(
  problem: ProblemSpec,
  nTrajectories: number,
  options: SampleMorrisOptions = {},
): MorrisDesign {
  const {
    numLevels = 4,
    seed = 0,
    verbose = true,
    method = "trajectory",
    truncationQuantile = 1e-4,
  } = options;

  if (!(truncationQuantile > 0 && truncationQuantile < 0.5)) {
    throw new Error(
      `jaxgsa.morris.sample: truncationQuantile must be in (0, 0.5), got ${truncationQuantile}`,
    );
  }

  if (nTrajectories < 2) {
    throw new Error(
      `jaxgsa.morris.sample: n_trajectories must be >= 2, got ${nTrajectories}`,
    );
  }

  if (numLevels < 2) {
    throw new Error(`jaxgsa.morris.sample: num_levels must be >= 2, got ${numLevels}`);
  }

  if (method === "radial") {
    throw new Error(
      "jaxgsa.morris.sample: method='radial' is not yet ported (only the " +
        "trajectory Morris design is implemented)",
    );
  }

  if (method !== "trajectory") {
    throw new Error(`jaxgsa.morris.sample: method must be "trajectory", got ${method}`);
  }

  if (numLevels % 2 !== 0) {
    console.warn(
      `jaxgsa.morris.sample: num_levels=${numLevels} is odd — grid levels are not ` +
        "equally probable and steps land off-grid; an even value is recommended",
    );
  }

  validateProblem(problem, "morris");
  const D = problem.names.length;

  const rng = splitmix32(seed >>> 0);

  const { expandedUnit, eeIdxAfter, eeIdxBefore, eeDelta } = buildTrajectories(
    nTrajectories,
    D,
    numLevels,
    rng,
  );

  squashOpenSides(expandedUnit, eeDelta, problem, D, truncationQuantile);

  // Deduplicate on the unit cube first (bitwise row equality), then transform.
  const { samples, expandedToUnique, nExpanded } = dedupeDesign(problem, expandedUnit);
  const nRuns = samples.length / D;

  if (verbose) {
    const duplicatesRemoved = nExpanded - nRuns;
    const duplicateFraction = nExpanded > 0 ? duplicatesRemoved / nExpanded : 0;
    console.info(
      `jaxgsa.morris.sample: D=${D}, method=trajectory, ` +
        `n_trajectories=${nTrajectories}, num_levels=${numLevels}, ` +
        `n_expanded=${nExpanded}, n_runs=${nRuns}, ` +
        `duplicates_removed=${duplicatesRemoved} (${(duplicateFraction * 100).toFixed(1)}%)`,
    );
  }

  return {
    samples,
    expandedToUnique,
    nExpanded,
    nTrajectories,
    numLevels,
    eeIdxAfter,
    eeIdxBefore,
    eeDelta,
    nParams: D,
  };
}