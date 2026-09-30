/**
 * Borgonovo delta port (scalar output): the given-data moment-independent
 * importance measure plus the given-data first-order Sobol S1.
 *
 * Port of `src/jaxgsa/borgonovo/_analyze.py` restricted to the scalar-output
 * path with `n_bootstrap = 0` (the default): one replicate (the original
 * sample), the Plischke class-count heuristic, a Gaussian KDE on a fixed
 * output grid, and the trapezoid L1 integral that defines delta. The Python
 * kernel casts everything to float64 when x64 is enabled (as the golden
 * generator does), so this port stays in host float64 throughout.
 *
 * Control flow (per-replicate scan, per-class KDE over grid tiles) is host
 * JS loops, the established replacement for `jax.lax.scan`/`map`; the math
 * per tile is elementwise on Float64Arrays.
 */

import type { ProblemSpec } from "@/jaxgsa/sampling";

const SQRT_2PI = Math.sqrt(2 * Math.PI);

const DEGENERATE_BW_TOL = 1e-2;

const DEGENERATE_BW_FRACTION = 0.1;

const MAX_CLASSES = 48;

const DEFAULT_GRID_SIZE = 100;

export interface BorgonovoOptions {
  /** Number of output-grid points for the KDE (default 100). */
  gridSize?: number;
  /** Bandwidth factor; `null` (default) applies the per-class Silverman rule. */
  bandwidth?: number | null;
  /** Class count floor control; `null` (default) uses the Plischke heuristic. */
  nClasses?: number | null;
}

export interface BorgonovoIndices {
  /** Borgonovo delta per parameter, (D,). */
  delta: Float64Array;
  /** Given-data first-order Sobol S1 per parameter, (D,). */
  S1: Float64Array;
}

/** Silverman rule `(0.75 * counts)**(-0.2) * std` (`_kde_bandwidths`). */
function kdeBandwidth(counts: number, std: number, factor: number | null): number {
  const f = factor ?? Math.pow(0.75 * counts, -0.2);

  return f * std;
}

/** Unconditional Gaussian KDE over the grid (`_kde_full`). */
function kdeFull(
  y: Float64Array,
  grid: Float64Array,
  h: number,
): Float64Array {
  const N = y.length;
  const G = grid.length;
  const safeH = h > 0 ? h : 1.0;
  const out = new Float64Array(G);
  const invNorm = 1 / (N * safeH * SQRT_2PI);

  for (let g = 0; g < G; g++) {
    let s = 0;

    for (let i = 0; i < N; i++) {
      const u = (grid[g] - y[i]) / safeH;
      s += Math.exp(-0.5 * u * u);
    }

    out[g] = h > 0 ? s * invNorm : 0.0;
  }

  return out;
}

/** Conditional Gaussian KDE over masked class members (`_kde` + mask). */
function kdeConditional(
  grid: Float64Array,
  members: Float64Array,
  h: number,
  count: number,
): Float64Array {
  const G = grid.length;
  const P = members.length;
  const safeH = h > 0 ? h : 1.0;
  const safeCount = Math.max(count, 1);
  const out = new Float64Array(G);
  const invNorm = 1 / (safeCount * safeH * SQRT_2PI);

  for (let g = 0; g < G; g++) {
    let s = 0;

    for (let p = 0; p < P; p++) {
      const u = (grid[g] - members[p]) / safeH;
      s += Math.exp(-0.5 * u * u);
    }

    out[g] = h > 0 ? s * invNorm : 0.0;
  }

  return out;
}

/** Composite trapezoid rule over the grid (`jnp.trapezoid` with `x=grid`). */
function trapezoid(f: Float64Array, grid: Float64Array): number {
  let acc = 0;

  for (let g = 1; g < grid.length; g++) {
    acc += (grid[g] - grid[g - 1]) * (f[g - 1] + f[g]);
  }

  return acc / 2;
}

/** Plischke heuristic class count (`_plischke_n_classes`). */
function plischkeNClasses(N: number): number {
  const exponent = 2 / (7 + Math.tanh((1500 - N) / 500));

  return Math.min(Math.ceil(Math.pow(N, exponent)), MAX_CLASSES);
}

/** Class layout: edges, per-class sizes, and the `take` index table. */
function classLayout(N: number, M: number) {
  const edges = Array.from({ length: M + 1 }, (_, m) => Math.floor((m / M) * N));

  const sizes = Array.from({ length: M }, (_, m) => edges[m + 1] - edges[m]);

  const nPad = Math.max(...sizes);
  const take = new Int32Array(M * nPad);

  for (let m = 0; m < M; m++) {
    for (let p = 0; p < nPad; p++) {
      take[m * nPad + p] = Math.min(edges[m] + p, N - 1);
    }
  }

  return { sizes, nPad, take };
}

/** Sample standard deviation, ddof=1 (`jnp.std` with `ddof=1`). */
function stdDdof1(values: Float64Array): number {
  const N = values.length;

  if (N < 2) return 0;

  let mean = 0;

  for (let i = 0; i < N; i++) mean += values[i];
  mean /= N;

  let s = 0;

  for (let i = 0; i < N; i++) {
    const d = values[i] - mean;
    s += d * d;
  }

  return Math.sqrt(s / (N - 1));
}

/** Population variance, ddof=0. */
function varPop(values: Float64Array): number {
  const N = values.length;

  if (N === 0) return 0;

  let mean = 0;

  for (let i = 0; i < N; i++) mean += values[i];
  mean /= N;

  let s = 0;

  for (let i = 0; i < N; i++) {
    const d = values[i] - mean;
    s += d * d;
  }

  return s / N;
}

/**
 * Borgonovo delta + given-data S1 for a scalar output. `x` is (N, D)
 * row-major; `y` is (N,). Mirrors `jaxgsa.borgonovo.analyze` with the
 * defaults (`n_bootstrap=0`, `grid_size=100`, Silverman bandwidth,
 * `degenerate_tol=1e-2`, `degenerate_bandwidth="auto"`).
 */
export function analyzeBorgonovo(
  problem: ProblemSpec,
  x: Float64Array | number[],
  y: Float64Array | number[],
  options: BorgonovoOptions = {},
): BorgonovoIndices {
  const xf = x instanceof Float64Array ? x : Float64Array.from(x);
  const yf = y instanceof Float64Array ? y : Float64Array.from(y);
  const D = problem.names.length;
  const N = xf.length / D;

  if (!Number.isInteger(N) || N === 0) {
    throw new Error(
      `borgonovo: x length ${xf.length} is not a multiple of D=${D}`,
    );
  }

  if (yf.length !== N) {
    throw new Error(`borgonovo: x has ${N} rows but y has ${yf.length} values`);
  }

  const gridSize = options.gridSize ?? DEFAULT_GRID_SIZE;
  const bwFactor = options.bandwidth ?? null;
  const M = options.nClasses ?? plischkeNClasses(N);

  // Constant inputs are forced to exactly 0 (mirrors the Python warn+zero).
  const constInputs = new Uint8Array(D);

  for (let d = 0; d < D; d++) {
    let mn = xf[d];
    let mx = xf[d];

    for (let i = 1; i < N; i++) {
      const v = xf[i * D + d];

      if (v < mn) mn = v;

      if (v > mx) mx = v;
    }

    constInputs[d] = mn === mx ? 1 : 0;
  }

  // Output grid over [y_min, y_max].
  let yMin = yf[0];
  let yMax = yf[0];

  for (let i = 1; i < N; i++) {
    if (yf[i] < yMin) yMin = yf[i];

    if (yf[i] > yMax) yMax = yf[i];
  }

  const grid = new Float64Array(gridSize);
  const span = yMax - yMin;

  for (let g = 0; g < gridSize; g++) grid[g] = yMin + (g / (gridSize - 1)) * span;

  // Class partition (single continuous group).
  const { sizes, nPad, take } = classLayout(N, M);

  // Per-column class membership: argsort X[:, d], then take table.
  const clsIdx = new Int32Array(D * M * nPad);

  for (let d = 0; d < D; d++) {
    const order = Array.from({ length: N }, (_, i) => i);
    order.sort((a, b) => xf[a * D + d] - xf[b * D + d]);

    for (let m = 0; m < M; m++) {
      for (let p = 0; p < nPad; p++) {
        clsIdx[(d * M + m) * nPad + p] = order[take[m * nPad + p]];
      }
    }
  }

  const delta = new Float64Array(D);
  const s1 = new Float64Array(D);

  // Full-sample stats.
  const yMean = (() => {
    let s = 0;

    for (let i = 0; i < N; i++) s += yf[i];

    return s / N;
  })();

  const yVar = varPop(yf);
  const safeVar = yVar > 0 ? yVar : 1.0;
  const hFull = kdeBandwidth(N, stdDdof1(yf), bwFactor);
  const fy = kdeFull(yf, grid, hFull);

  const gridStep = gridSize > 1 ? grid[1] - grid[0] : 0;
  const degenerateFloor = Math.max(DEGENERATE_BW_FRACTION * hFull, gridStep);

  for (let d = 0; d < D; d++) {
    if (constInputs[d]) {
      delta[d] = 0;
      s1[d] = 0;

      continue;
    }

    let deltaSum = 0;
    let viSum = 0;

    for (let m = 0; m < M; m++) {
      const count = sizes[m];
      const safeCount = Math.max(count, 1);
      const members = new Float64Array(nPad);
      let valid = 0;

      for (let p = 0; p < nPad; p++) {
        if (p < count) {
          members[p] = yf[clsIdx[(d * M + m) * nPad + p]];
          valid++;
        }
      }

      // Class mean/variance, ddof=1 over valid members (`_group_stats`).
      let clsMean = 0;

      for (let p = 0; p < valid; p++) clsMean += members[p];
      clsMean /= Math.max(valid, 1);

      let clsVar = 0;

      for (let p = 0; p < valid; p++) {
        const dv = members[p] - clsMean;
        clsVar += dv * dv;
      }

      clsVar /= Math.max(count - 1, 1);

      const h = kdeBandwidth(safeCount, Math.sqrt(clsVar), bwFactor);
      const floored = count > 0 && h < DEGENERATE_BW_TOL * hFull;
      const effH = floored ? degenerateFloor : h;
      const fyc = kdeConditional(grid, members.subarray(0, valid), effH, count);

      // L1 integral of |fy - fyc| over the grid.
      const l1arr = new Float64Array(gridSize);

      for (let g = 0; g < gridSize; g++) l1arr[g] = Math.abs(fy[g] - fyc[g]);

      const l1 = trapezoid(l1arr, grid);
      deltaSum += (count / (2 * N)) * l1;

      const clsMeanDev = clsMean - yMean;
      viSum += (count / N) * clsMeanDev * clsMeanDev;
    }

    delta[d] = deltaSum;
    s1[d] = yVar > 0 ? viSum / safeVar : 0.0;
  }

  return { delta, S1: s1 };
}