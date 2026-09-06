import { sobolSequence } from "./sampler";
import {
  stableUniqueRows,
  transformSamples,
  type ProblemSpec,
} from "../sampling";

/**
 * Port of `jaxgsa.sobol.sample` and its helpers
 * (`_build_expanded_samples`, `_stable_unique_rows`, `_transform_samples`)
 * to plain TS. No jax-js involved: the Python sampler is host numpy and the
 * sampling step is cheap; wasm/jax-js compute comes later in analyze.
 *
 * The marginal types, the inverse-CDF transforms and the stable row
 * deduplication now live in `../sampling.ts` (shared with the Morris port);
 * they are re-exported below so existing imports keep working.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export { type MarginalSpec, type ProblemSpec } from "../sampling";

export interface SobolDesign {
  /** Unique rows in physical units, shape (nRuns, nParams), row-major. */
  samples: Float64Array;
  /** Rows of the expanded Saltelli design before deduplication. */
  nExpanded: number;
  /** Map from each expanded row to its retained unique row. */
  expandedToUnique: Int32Array;
  baseN: number;
  nParams: number;
  calcSecondOrder: boolean;
}

// ---------------------------------------------------------------------------
// Saltelli layout (`_saltelli_step`, `_build_expanded_samples`)
// ---------------------------------------------------------------------------

/** Rows of the expanded Saltelli design per base Sobol' point. */
export function saltelliStep(nParams: number, calcSecondOrder: boolean): number {
  return calcSecondOrder ? 2 * nParams + 2 : nParams + 2;
}

export interface ExpandedOptions {
  calcSecondOrder: boolean;
  scramble: boolean;
  seed: number;
}

/**
 * Draw `baseN` points from a 2D-dimensional Sobol' sequence (first D columns
 * = matrix A, last D = matrix B) and build the Saltelli interleaved layout:
 * per base point i the rows are
 *   [A_i, AB_0, ..., AB_{D-1}, BA_0, ..., BA_{D-1}, B_i]
 * with AB_j = A with column j replaced by B's column j (and BA_j the
 * transpose), the BA block present only for second order. Output is
 * (baseN * step, D) row-major. Exact duplicate rows from the Saltelli
 * collapse are kept here and removed later by deduplication.
 */
export function buildExpandedSamples(
  nParams: number,
  baseN: number,
  options: ExpandedOptions,
): Float64Array {
  const D = nParams;
  const step = saltelliStep(D, options.calcSecondOrder);
  const base = sobolSequence(2 * D, baseN, options.scramble, options.seed);
  const out = new Float64Array(baseN * step * D);
  for (let i = 0; i < baseN; i++) {
    const b = i * 2 * D;
    const o = i * step * D;
    // A_i
    for (let j = 0; j < D; j++) out[o + j] = base[b + j];
    // AB_0..AB_{D-1}
    for (let jj = 0; jj < D; jj++) {
      const oo = o + (1 + jj) * D;
      for (let j = 0; j < D; j++) out[oo + j] = base[b + j];
      out[oo + jj] = base[b + D + jj];
    }
    // BA_0..BA_{D-1} (second order only)
    if (options.calcSecondOrder) {
      for (let jj = 0; jj < D; jj++) {
        const oo = o + (1 + D + jj) * D;
        for (let j = 0; j < D; j++) out[oo + j] = base[b + D + j];
        out[oo + jj] = base[b + jj];
      }
    }
    // B_i
    for (let j = 0; j < D; j++) out[o + (step - 1) * D + j] = base[b + D + j];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Shared helpers (moved to `../sampling.ts`, re-exported for compatibility)
// ---------------------------------------------------------------------------

export {
  UNIT_CLIP,
  normalInverseCdf,
  erfComplement,
  normalCdf,
  truncatedGaussianInverse,
  transformValue,
  transformSamples,
  stableUniqueRows,
} from "../sampling";

/** Deduplicate on the unit cube first, then transform (`_dedupe_design`). */
export function dedupeDesign(
  problem: ProblemSpec,
  expandedUnit: Float64Array,
): { samples: Float64Array; expandedToUnique: Int32Array; nExpanded: number } {
  const D = problem.names.length;
  const { unique, expandedToUnique } = stableUniqueRows(expandedUnit, D);
  const samples = transformSamples(problem, unique);
  return { samples, expandedToUnique, nExpanded: expandedUnit.length / D };
}

// ---------------------------------------------------------------------------
// Public entry point (`sample`)
// ---------------------------------------------------------------------------

export const MAX_BASE_N = 1 << 24;

function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

function nextPowerOfTwo(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

export interface SampleOptions {
  /** Exact Sobol' base size; must be a power of 2. */
  baseN?: number;
  /** Include the BA cross-matrices needed for second-order indices (default true). */
  calcSecondOrder?: boolean;
  /** Apply the seeded affine scramble (default true). */
  scramble?: boolean;
  seed?: number;
  verbose?: boolean;
}

/**
 * Generate the input samples needed for Sobol' analysis: a Saltelli design
 * over a candidate `baseN`, deduplicated to unique rows in first-occurrence
 * order, doubling `baseN` until the requested number of unique rows exists.
 * Mirrors `jaxgsa.sobol.sample` (continuous problems only for now).
 */
export function sample(
  problem: ProblemSpec,
  nSamples: number,
  options: SampleOptions = {},
): SobolDesign {
  const {
    baseN: baseNGiven,
    calcSecondOrder = true,
    scramble = true,
    seed = 0,
    verbose = true,
  } = options;

  if (problem.correlation != null) {
    throw new Error(
      "jaxgsa.sobol.sample: correlated problems are not supported (the " +
        "Saltelli design and its estimators assume independent inputs)",
    );
  }
  const D = problem.names.length;
  if (D < 1) {
    throw new Error("jaxgsa.sobol.sample: problem must declare at least one parameter");
  }
  if (D > 20) {
    throw new Error(
      `jaxgsa.sobol.sample: at most 20 parameters supported (a Saltelli design ` +
        `needs 2D = ${2 * D} Sobol' dimensions, capped at 40)`,
    );
  }
  if (problem.marginals.length !== D) {
    throw new Error(
      `jaxgsa.sobol.sample: problem declares ${D} names but ${problem.marginals.length} marginals`,
    );
  }

  const step = saltelliStep(D, calcSecondOrder);
  let baseN: number;
  let targetN: number | null;
  if (baseNGiven !== undefined) {
    if (!isPowerOfTwo(baseNGiven)) {
      throw new Error(
        `jaxgsa.sobol.sample: base_n must be a power of 2 (got ${baseNGiven})`,
      );
    }
    baseN = baseNGiven;
    targetN = null;
  } else {
    targetN = Math.max(1, nSamples);
    baseN = nextPowerOfTwo(Math.ceil(targetN / step));
    if (baseN > MAX_BASE_N) {
      throw new Error(
        `jaxgsa.sobol.sample: n_samples=${targetN} needs base_n=${baseN}, above ` +
          `the cap of 2^24 = ${MAX_BASE_N}`,
      );
    }
  }

  if (baseN < 16) {
    console.warn(
      `jaxgsa.sobol.sample: base_n=${baseN} is below the floor of 16 base ` +
        "points. The design is degenerate: the Sobol' indices will be " +
        "unreliable. Raise n_samples (or base_n) for a usable design.",
    );
  }

  const build = () =>
    dedupeDesign(problem, buildExpandedSamples(D, baseN, { calcSecondOrder, scramble, seed }));

  let design = build();
  let nExpanded = design.nExpanded;

  if (targetN !== null) {
    let doublings = 0;
    while (design.samples.length / D < targetN) {
      if (doublings >= 32 || baseN >= MAX_BASE_N) {
        console.warn(
          `jaxgsa.sobol.sample: n_samples=${targetN} unique rows cannot be ` +
            `reached${baseN >= MAX_BASE_N ? " (base_n cap of 2^24 hit)" : ""} after ` +
            `${doublings} doublings; returning ${design.samples.length / D} unique rows ` +
            "instead (duplicates are valid Saltelli samples)",
        );
        break;
      }
      baseN *= 2;
      doublings += 1;
      design = build();
      nExpanded = design.nExpanded;
    }
  }

  const nRuns = design.samples.length / D;
  if (verbose) {
    const duplicatesRemoved = nExpanded - nRuns;
    const duplicateFraction = nExpanded > 0 ? duplicatesRemoved / nExpanded : 0;
    console.info(
      `jaxgsa.sobol.sample: D=${D}, mode=${calcSecondOrder ? "second-order" : "first/total-order"}, ` +
        `base_n=${baseN}, n_runs=${nRuns}, n_expanded=${nExpanded}, ` +
        `duplicates_removed=${duplicatesRemoved} (${(duplicateFraction * 100).toFixed(1)}%), ` +
        `scramble=${scramble}`,
    );
  }

  return {
    samples: design.samples,
    nExpanded,
    expandedToUnique: design.expandedToUnique,
    baseN,
    nParams: D,
    calcSecondOrder,
  };
}