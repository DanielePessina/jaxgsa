/**
 * The site engine: jax-js device bootstrap, CSV/run_id exchange helpers, and
 * the thin typed wrappers that wire the React layer to the method ports.
 *
 * Ownership rule (see PLAN-WASM.md): every port call consumes its input
 * arrays. The analyze ports build their own device copies from the host
 * arrays they receive, and `analyzeKucherenko` additionally consumes the
 * `np.Array` objects this module builds — so each workflow call creates
 * fresh arrays and never reuses a device buffer across calls.
 */

import { defaultDevice, init, numpy as np } from "@jax-js/jax";
import { ishigami } from "@/jaxgsa/benchmarks";
import { analyzeKucherenko } from "@/jaxgsa/kucherenko";
import { sampleKucherenkoDesign, type KucherenkoDesign } from "@/jaxgsa/kucherenko/sample";
import { analyzeMorris, type MorrisMeasures } from "@/jaxgsa/morris/analyze";
import { sampleMorris, type MorrisDesign } from "@/jaxgsa/morris/sample";
import { analyzePce, type PceIndices } from "@/jaxgsa/pce/analyze";
import { analyzeSobol, type SobolIndices } from "@/jaxgsa/sobol/analyze";
import { sample, type SobolDesign } from "@/jaxgsa/sobol/sample";
import { type ProblemSpec } from "@/jaxgsa/sampling";
import { analyzeShapleyPce, type ShapleyIndices } from "@/jaxgsa/shapley/analyze";

// ---------------------------------------------------------------------------
// Device bootstrap
// ---------------------------------------------------------------------------

export interface DeviceInfo {
  device: string;
  webgpuAvailable: boolean;
}

let engineInit: Promise<DeviceInfo> | null = null;

/**
 * Initialize the jax-js runtime and pin the compute device to `wasm`
 * (float64). WebGPU exists only as a detection flag: float32 acceleration is
 * a future step, and float64 exactness is the design decision.
 */
export function initEngine(): Promise<DeviceInfo> {
  if (engineInit === null) {
    engineInit = (async () => {
      const devices = await init();
      defaultDevice("wasm");
      return { device: "wasm", webgpuAvailable: devices.includes("webgpu") };
    })();
  }
  return engineInit;
}

// ---------------------------------------------------------------------------
// Demo problem
// ---------------------------------------------------------------------------

/** Ishigami over [-pi, pi]^3 — the only problem for now (read-only demo). */
export const ISHIGAMI_PROBLEM: ProblemSpec = {
  names: ["x1", "x2", "x3"],
  marginals: [
    { kind: "uniform", low: -Math.PI, high: Math.PI },
    { kind: "uniform", low: -Math.PI, high: Math.PI },
    { kind: "uniform", low: -Math.PI, high: Math.PI },
  ],
};

// ---------------------------------------------------------------------------
// CSV helpers
// ---------------------------------------------------------------------------

export interface ParsedCsv {
  headers: string[];
  rows: number[][];
}

/**
 * Parse a plain CSV: one header row, comma-separated numbers, tolerant of
 * trailing newlines and blank lines. Throws on non-numeric cells.
 */
export function parseCsv(text: string): ParsedCsv {
  const lines = text.split(/\r?\n/);
  while (lines.length > 0 && lines[lines.length - 1].trim() === "") lines.pop();
  if (lines.length === 0) return { headers: [], rows: [] };

  const headers = lines[0].split(",").map((s) => s.trim());
  const rows: number[][] = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === "") continue;
    const cells = line.split(",").map((s) => s.trim());
    const row: number[] = [];
    for (let j = 0; j < cells.length; j++) {
      if (cells[j] === "") {
        throw new Error(`line ${i + 1}: empty cell in column ${j + 1}`);
      }
      const v = Number(cells[j]);
      if (Number.isNaN(v)) {
        throw new Error(`line ${i + 1}: "${cells[j]}" is not a number`);
      }
      row.push(v);
    }
    rows.push(row);
  }
  return { headers, rows };
}

/**
 * Serialize a design to CSV: a `run_id` column (0..n_runs-1, the canonical
 * row order) followed by one parameter column per problem marginal.
 */
export function buildDesignCsv(
  problem: ProblemSpec,
  design: { samples: Float64Array },
): string {
  const D = problem.names.length;
  const nRuns = design.samples.length / D;
  const lines = [`run_id,${problem.names.join(",")}`];
  for (let r = 0; r < nRuns; r++) {
    const cells = [String(r)];
    for (let j = 0; j < D; j++) cells.push(String(design.samples[r * D + j]));
    lines.push(cells.join(","));
  }
  return lines.join("\n") + "\n";
}

/**
 * The run_id join: `runIds[i]` is the run_id of `yRows[i]`; the returned
 * array has output `runIds[i]` written at position `runIds[i]`, i.e. the
 * design's canonical row order. Validates that the ids form a permutation of
 * 0..n_runs-1 (right count, no duplicates, nothing missing) and that every
 * row carries at least one output value (the first column is the scalar
 * output).
 */
export function alignYByRunId(
  yRows: number[][],
  runIds: number[],
  design: { samples: Float64Array; nParams: number },
): Float64Array {
  const nRuns = design.samples.length / design.nParams;
  if (!Number.isInteger(nRuns)) {
    throw new Error("design.samples is not a whole number of rows");
  }
  if (yRows.length !== nRuns) {
    throw new Error(
      `run_id row count mismatch: the design has ${nRuns} rows but the ` +
        `uploaded Y CSV has ${yRows.length}`,
    );
  }
  if (runIds.length !== nRuns) {
    throw new Error(
      `run_id column mismatch: expected ${nRuns} ids, found ${runIds.length}`,
    );
  }

  const seen = new Uint8Array(nRuns);
  const out = new Float64Array(nRuns);
  for (let i = 0; i < nRuns; i++) {
    const id = runIds[i];
    if (!Number.isInteger(id) || id < 0 || id >= nRuns) {
      throw new Error(
        `line ${i + 2}: invalid run_id ${id} (expected an integer in [0, ${nRuns}))`,
      );
    }
    if (seen[id] === 1) {
      throw new Error(`line ${i + 2}: duplicate run_id ${id}`);
    }
    if (yRows[i].length < 1) {
      throw new Error(`line ${i + 2}: row for run_id ${id} has no output value`);
    }
    seen[id] = 1;
    out[id] = yRows[i][0];
  }
  for (let id = 0; id < nRuns; id++) {
    if (seen[id] === 0) {
      throw new Error(
        `missing run_id ${id}: the uploaded Y CSV must cover every design row`,
      );
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Design generation (workflow A)
// ---------------------------------------------------------------------------

export type DesignMethod = "sobol" | "morris" | "kucherenko";
export type GivenDataMethod = "pce" | "shapley";

export interface SobolConfig {
  baseN: number;
  calcSecondOrder: boolean;
  seed: number;
}
export interface MorrisConfig {
  nTrajectories: number;
  numLevels: number;
  seed: number;
}
export interface KucherenkoConfig {
  nSamples: number;
  seed: number;
}
export type DesignConfig = SobolConfig | MorrisConfig | KucherenkoConfig;

export type PortDesign = SobolDesign | MorrisDesign | KucherenkoDesign;

export interface GeneratedDesign {
  method: DesignMethod;
  problem: ProblemSpec;
  /** Rows in physical units, (nRuns, D) row-major; row index == run_id. */
  samples: Float64Array;
  nRuns: number;
  nParams: number;
  /** The port object the analysis step consumes. */
  port: PortDesign;
  /** Key/value summary rows for the UI (base_n, trajectories, ...). */
  summary: [string, string][];
  notes: string[];
}

/**
 * Generate a design for the given method. The Sobol' port cannot estimate
 * second-order indices yet, so a `calcSecondOrder` request degrades to the
 * first/total-order layout with a note (the S2 estimator is not ported).
 */
export function generateDesign(
  method: DesignMethod,
  problem: ProblemSpec,
  config: DesignConfig,
): GeneratedDesign {
  const D = problem.names.length;
  const notes: string[] = [];

  if (method === "sobol") {
    const cfg = config as SobolConfig;
    if (cfg.calcSecondOrder) {
      notes.push(
        "S2 is not ported yet — the design was generated in first/total-order layout.",
      );
    }
    const port = sample(problem, 0, {
      baseN: cfg.baseN,
      calcSecondOrder: false,
      seed: cfg.seed,
      verbose: false,
    });
    const nRuns = port.samples.length / D;
    return {
      method,
      problem,
      samples: port.samples,
      nRuns,
      nParams: D,
      port,
      summary: [
        ["base_n", String(port.baseN)],
        ["n_expanded", String(port.nExpanded)],
        ["n_runs", String(nRuns)],
        [
          "duplicates removed",
          String(port.nExpanded - nRuns),
        ],
      ],
      notes,
    };
  }

  if (method === "morris") {
    const cfg = config as MorrisConfig;
    const port = sampleMorris(problem, cfg.nTrajectories, {
      numLevels: cfg.numLevels,
      seed: cfg.seed,
      verbose: false,
    });
    const nRuns = port.samples.length / D;
    return {
      method,
      problem,
      samples: port.samples,
      nRuns,
      nParams: D,
      port,
      summary: [
        ["trajectories", String(port.nTrajectories)],
        ["grid levels", String(port.numLevels)],
        ["n_expanded", String(port.nExpanded)],
        ["n_runs", String(nRuns)],
      ],
      notes,
    };
  }

  const cfg = config as KucherenkoConfig;
  const port = sampleKucherenkoDesign(problem, cfg.nSamples, cfg.seed);
  const nRuns = port.samples.length / D;
  return {
    method,
    problem,
    samples: port.samples,
    nRuns,
    nParams: D,
    port,
    summary: [
      ["base_n", String(port.baseN)],
      ["blocks (2D+1)", String(2 * D + 1)],
      ["n_runs", String(nRuns)],
    ],
    notes,
  };
}

// ---------------------------------------------------------------------------
// Workflow A analysis
// ---------------------------------------------------------------------------

export interface ResultColumn {
  key: string;
  label: string;
  values: Float64Array;
}

export interface AnalysisResult {
  method: string;
  parameters: string[];
  columns: ResultColumn[];
  notes: string[];
}

function result(
  method: string,
  parameters: string[],
  columns: ResultColumn[],
  notes: string[],
): AnalysisResult {
  return { method, parameters, columns, notes };
}

/** Evaluate the Ishigami benchmark at every design row (nRuns outputs). */
export function evaluateIshigami(design: {
  samples: Float64Array;
  nParams: number;
}): Float64Array {
  const D = design.nParams;
  const nRuns = design.samples.length / D;
  const y = new Float64Array(nRuns);
  const row: number[] = new Array(D);
  for (let r = 0; r < nRuns; r++) {
    for (let j = 0; j < D; j++) row[j] = design.samples[r * D + j];
    y[r] = ishigami(row);
  }
  return y;
}

/**
 * Run the analysis for a generated design. `y` carries one output per design
 * row in run_id order (use `alignYByRunId` after an upload). Fresh device
 * arrays are built per port call; nothing is reused.
 */
export function analyzeGenerated(
  gen: GeneratedDesign,
  y: Float64Array | number[],
): AnalysisResult {
  const params = gen.problem.names;

  if (gen.method === "sobol") {
    const design = gen.port as SobolDesign;
    const { S1, ST }: SobolIndices = analyzeSobol(y, design.nParams, {
      expandedToUnique: design.expandedToUnique,
    });
    return result(gen.method, params, [
      { key: "S1", label: "S1", values: S1 },
      { key: "ST", label: "ST", values: ST },
    ], gen.notes);
  }

  if (gen.method === "morris") {
    const design = gen.port as MorrisDesign;
    const { mu, mu_star, sigma }: MorrisMeasures = analyzeMorris(design, y);
    return result(gen.method, params, [
      { key: "mu", label: "mu", values: mu },
      { key: "mu_star", label: "mu*", values: mu_star },
      { key: "sigma", label: "sigma", values: sigma },
    ], gen.notes);
  }

  const design = gen.port as KucherenkoDesign;
  const nRuns = design.samples.length / design.nParams;
  const xNp = np
    .array(design.samples as Float64Array<ArrayBuffer>, { dtype: np.float64 })
    .reshape([nRuns, design.nParams]);
  const yNp = np.array(y as Float64Array<ArrayBuffer>, { dtype: np.float64 });
  const { S1, ST } = analyzeKucherenko(xNp, yNp);
  return result(gen.method, params, [
    { key: "S1", label: "S1", values: S1 },
    { key: "ST", label: "ST", values: ST },
  ], gen.notes);
}

// ---------------------------------------------------------------------------
// Workflow B: given data (surrogates)
// ---------------------------------------------------------------------------

/**
 * Run a PCE or Shapley(PCE) analysis on an arbitrary (X, Y) point cloud.
 * `x` is (N, D) row-major. Each port call builds its own device arrays from
 * the host inputs.
 */
export function analyzeCloud(
  method: GivenDataMethod,
  problem: ProblemSpec,
  x: Float64Array | number[],
  y: Float64Array | number[],
  order: number,
): AnalysisResult {
  const params = problem.names;
  const xf = x instanceof Float64Array ? x : Float64Array.from(x);
  const yf = y instanceof Float64Array ? y : Float64Array.from(y);
  const N = xf.length / problem.names.length;
  if (!Number.isInteger(N)) {
    throw new Error(
      `x has ${xf.length} values, not a multiple of D=${problem.names.length}`,
    );
  }
  if (yf.length !== N) {
    throw new Error(`x has ${N} rows but y has ${yf.length} values`);
  }

  if (method === "pce") {
    const { S1, ST }: PceIndices = analyzePce(problem, xf, yf, { order });
    return result(method, params, [
      { key: "S1", label: "S1", values: S1 },
      { key: "ST", label: "ST", values: ST },
    ], []);
  }

  const { Sh, S1, ST }: ShapleyIndices = analyzeShapleyPce(problem, xf, yf, {
    order,
  });
  return result(method, params, [
    { key: "Sh", label: "Sh", values: Sh },
    { key: "S1", label: "S1", values: S1 },
    { key: "ST", label: "ST", values: ST },
  ], []);
}

/** Ishigami point cloud for the "load example" path of workflow B. */
export function loadExampleCloud(
  nSamples = 1024,
  seed = 0,
): { x: Float64Array; y: Float64Array; n: number } {
  const port = sample(ISHIGAMI_PROBLEM, nSamples, {
    calcSecondOrder: false,
    seed,
    verbose: false,
  });
  const x = port.samples;
  const y = evaluateIshigami({ samples: x, nParams: port.nParams });
  return { x, y, n: x.length / port.nParams };
}