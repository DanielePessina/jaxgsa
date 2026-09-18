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
import { analyzeKucherenko } from "@/jaxgsa/kucherenko";
import { sampleKucherenkoDesign, type KucherenkoDesign } from "@/jaxgsa/kucherenko/sample";
import { analyzeMorris, type MorrisMeasures } from "@/jaxgsa/morris/analyze";
import { sampleMorris, type MorrisDesign } from "@/jaxgsa/morris/sample";
import { analyzePce, type PceIndices } from "@/jaxgsa/pce/analyze";
import { analyzeSobol, type SobolIndices } from "@/jaxgsa/sobol/analyze";
import { sample, type SobolDesign } from "@/jaxgsa/sobol/sample";
import { type ProblemSpec } from "@/jaxgsa/sampling";
import { analyzeShapleyPce, type ShapleyIndices } from "@/jaxgsa/shapley/analyze";
import type { Demo } from "./demos";

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

// ---------------------------------------------------------------------------
// Y exchange model (PLAN-WEB-UI.md D1)
// ---------------------------------------------------------------------------

/** One output column of a Y upload: an output at a single time point. */
export interface YColumn {
  output: string;
  time: number;
  /** One value per run, in upload row order (unaligned). */
  values: Float64Array;
}

/** The normalized Y upload: every (output, time) slice of the model outputs. */
export interface YData {
  columns: YColumn[];
  rowCount: number;
  label: string;
}

/** The `_t{t}` timepoint suffix: `<output>_t<digits>`. */
const TIME_SUFFIX_RE = /^(.+)_t([0-9]+)$/;

/**
 * Classify one output column name per the `_t{t}` exchange contract:
 * `<output>_t<digits>` names that output at a time point; anything else is
 * `<name>` at time 0. `_t<digits>` is a reserved suffix — an output literally
 * named e.g. `foo_t2` must be written `foo_t2_t0`. See PLAN-WEB-UI.md D1.
 */
export function classifyYColumn(name: string): { output: string; time: number } {
  const m = TIME_SUFFIX_RE.exec(name);
  if (m) return { output: m[1], time: Number(m[2]) };
  return { output: name, time: 0 };
}

/** Stable display label for a parsed column (round-trips the source header). */
export function yColumnLabel(output: string, time: number): string {
  return time === 0 ? output : `${output}_t${time}`;
}

/**
 * Classify a parsed CSV into output columns per the contract. `run_id` is
 * extracted (not a data column) and columns whose header matches a problem
 * parameter name are ignored — so a user may append outputs to the downloaded
 * design CSV and upload the whole file. Rejects rows that do not match the
 * header width and duplicate (output, time) columns.
 */
export function parseYColumns(
  parsed: ParsedCsv,
  problem: ProblemSpec,
): { columns: YColumn[]; runIds: number[] | null; rowCount: number } {
  const headers = parsed.headers.map((h) => h.trim());
  const xCols = new Set(problem.names);
  let runIdIdx = -1;
  for (let j = 0; j < headers.length; j++) {
    if (headers[j] === "run_id") {
      if (runIdIdx >= 0) throw new Error("duplicate run_id column");
      runIdIdx = j;
    }
  }

  const defs: { header: string; output: string; time: number; src: number }[] = [];
  for (let j = 0; j < headers.length; j++) {
    if (j === runIdIdx) continue;
    if (xCols.has(headers[j])) continue;
    defs.push({ header: headers[j], ...classifyYColumn(headers[j]), src: j });
  }
  if (defs.length === 0) {
    throw new Error(
      "no output columns found: the Y file needs at least one column " +
        "besides run_id and the parameter names",
    );
  }

  const seen = new Set<string>();
  for (const d of defs) {
    const key = `${d.output}\u0000${d.time}`;
    if (seen.has(key)) {
      throw new Error(
        `duplicate output column "${d.header}": ${d.output} at t${d.time} ` +
          `appears more than once`,
      );
    }
    seen.add(key);
  }

  const n = parsed.rows.length;
  for (let i = 0; i < n; i++) {
    if (parsed.rows[i].length !== headers.length) {
      throw new Error(
        `line ${i + 2}: expected ${headers.length} columns, found ${parsed.rows[i].length}`,
      );
    }
  }

  const columns: YColumn[] = defs.map((d) => {
    const values = new Float64Array(n);
    for (let i = 0; i < n; i++) values[i] = parsed.rows[i][d.src];
    return { output: d.output, time: d.time, values };
  });

  let runIds: number[] | null = null;
  if (runIdIdx >= 0) runIds = parsed.rows.map((r) => r[runIdIdx]);
  return { columns, runIds, rowCount: n };
}

/**
 * Wrap scalar values as the canonical single-output YData (demo path).
 */
export function scalarYData(values: Float64Array, label: string): YData {
  return {
    columns: [{ output: "y", time: 0, values }],
    rowCount: values.length,
    label,
  };
}

/** Short human summary of the slices a YData carries. */
export function describeY(y: YData): string {
  const outs = new Set(y.columns.map((c) => c.output));
  const maxT = Math.max(...y.columns.map((c) => c.time));
  const tDesc = maxT === 0 ? "1 time step" : `${maxT + 1} time steps`;
  return `${outs.size} output${outs.size > 1 ? "s" : ""} · ${tDesc} · ${y.rowCount} runs`;
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
 * The run_id join, generalized to every output column: `runIds[i]` is the
 * run_id of the upload row `i`; each returned column has output
 * `runIds[i]` written at position `runIds[i]`, i.e. the design's canonical
 * row order. Validates that the ids form a permutation of 0..n_runs-1
 * (right count, no duplicates, nothing missing).
 */
export function alignColumnsByRunId(
  columns: YColumn[],
  runIds: number[],
  design: { samples: Float64Array; nParams: number },
): YColumn[] {
  const nRuns = design.samples.length / design.nParams;
  if (!Number.isInteger(nRuns)) {
    throw new Error("design.samples is not a whole number of rows");
  }
  if (columns.length > 0 && columns[0].values.length !== nRuns) {
    throw new Error(
      `run_id row count mismatch: the design has ${nRuns} rows but the ` +
        `uploaded Y CSV has ${columns[0].values.length}`,
    );
  }
  if (runIds.length !== nRuns) {
    throw new Error(
      `run_id column mismatch: expected ${nRuns} ids, found ${runIds.length}`,
    );
  }

  const seen = new Uint8Array(nRuns);
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
    seen[id] = 1;
  }
  for (let id = 0; id < nRuns; id++) {
    if (seen[id] === 0) {
      throw new Error(
        `missing run_id ${id}: the uploaded Y CSV must cover every design row`,
      );
    }
  }

  return columns.map((c) => {
    const out = new Float64Array(nRuns);
    for (let i = 0; i < nRuns; i++) out[runIds[i]] = c.values[i];
    return { output: c.output, time: c.time, values: out };
  });
}

// ---------------------------------------------------------------------------
// Design generation (workflow A)
// ---------------------------------------------------------------------------

export type DesignMethod = "sobol" | "morris" | "kucherenko";
export type GivenDataMethod = "pce" | "shapley";

/** Where the X input comes from: a sampled design or an uploaded point cloud. */
export type XSource = { kind: "design"; method: DesignMethod } | { kind: "uploaded" };

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

/** One (output, time) slice of a result — the indices for that output column. */
export interface ResultSlice {
  output: string;
  time: number;
  columns: ResultColumn[];
}

export interface AnalysisResult {
  method: string;
  parameters: string[];
  slices: ResultSlice[];
  notes: string[];
  /** The settings the analysis ran with (seed, order, base_n, ...) for repro. */
  settings: Record<string, unknown>;
}

function result(
  method: string,
  parameters: string[],
  slices: ResultSlice[],
  notes: string[],
  settings: Record<string, unknown>,
): AnalysisResult {
  return { method, parameters, slices, notes, settings };
}

/**
 * Run the analysis for a generated design. `y` carries one or more output
 * columns in run_id order (use `parseYColumns` + `alignColumnsByRunId` after
 * an upload). Every (output, time) slice is analyzed independently through
 * the scalar ports. Fresh device arrays are built per port call; nothing is
 * reused.
 */
export function analyzeGenerated(gen: GeneratedDesign, y: YData): AnalysisResult {
  const params = gen.problem.names;
  const slices: ResultSlice[] = [];

  for (const col of y.columns) {
    if (gen.method === "sobol") {
      const design = gen.port as SobolDesign;
      const { S1, ST }: SobolIndices = analyzeSobol(col.values, design.nParams, {
        expandedToUnique: design.expandedToUnique,
      });
      slices.push({
        output: col.output,
        time: col.time,
        columns: [
          { key: "S1", label: "S1", values: S1 },
          { key: "ST", label: "ST", values: ST },
        ],
      });
    } else if (gen.method === "morris") {
      const design = gen.port as MorrisDesign;
      const { mu, mu_star, sigma }: MorrisMeasures = analyzeMorris(design, col.values);
      slices.push({
        output: col.output,
        time: col.time,
        columns: [
          { key: "mu", label: "mu", values: mu },
          { key: "mu_star", label: "mu*", values: mu_star },
          { key: "sigma", label: "sigma", values: sigma },
        ],
      });
    } else {
      const design = gen.port as KucherenkoDesign;
      const nRuns = design.samples.length / design.nParams;
      const xNp = np
        .array(design.samples as Float64Array<ArrayBuffer>, { dtype: np.float64 })
        .reshape([nRuns, design.nParams]);
      const yNp = np.array(col.values as Float64Array<ArrayBuffer>, {
        dtype: np.float64,
      });
      const { S1, ST } = analyzeKucherenko(xNp, yNp);
      slices.push({
        output: col.output,
        time: col.time,
        columns: [
          { key: "S1", label: "S1", values: S1 },
          { key: "ST", label: "ST", values: ST },
        ],
      });
    }
  }

  return result(gen.method, params, slices, gen.notes, {
    ...Object.fromEntries(gen.summary),
  });
}

// ---------------------------------------------------------------------------
// Workflow B: given data (surrogates)
// ---------------------------------------------------------------------------

/**
 * Run a PCE or Shapley(PCE) analysis on an arbitrary (X, Y) point cloud.
 * `x` is (N, D) row-major; `y` carries one or more output columns of length
 * N. Each (output, time) slice is analyzed independently. Each port call
 * builds its own device arrays from the host inputs.
 */
export function analyzeCloud(
  method: GivenDataMethod,
  problem: ProblemSpec,
  x: Float64Array | number[],
  y: YData,
  order: number,
): AnalysisResult {
  const params = problem.names;
  const xf = x instanceof Float64Array ? x : Float64Array.from(x);
  const N = xf.length / problem.names.length;
  if (!Number.isInteger(N)) {
    throw new Error(
      `x has ${xf.length} values, not a multiple of D=${problem.names.length}`,
    );
  }

  const slices: ResultSlice[] = [];
  for (const col of y.columns) {
    if (col.values.length !== N) {
      throw new Error(
        `x has ${N} rows but output "${col.output}" (t${col.time}) has ${col.values.length} values`,
      );
    }
    if (method === "pce") {
      const { S1, ST }: PceIndices = analyzePce(problem, xf, col.values, { order });
      slices.push({
        output: col.output,
        time: col.time,
        columns: [
          { key: "S1", label: "S1", values: S1 },
          { key: "ST", label: "ST", values: ST },
        ],
      });
    } else {
      const { Sh, S1, ST }: ShapleyIndices = analyzeShapleyPce(problem, xf, col.values, {
        order,
      });
      slices.push({
        output: col.output,
        time: col.time,
        columns: [
          { key: "Sh", label: "Sh", values: Sh },
          { key: "S1", label: "S1", values: S1 },
          { key: "ST", label: "ST", values: ST },
        ],
      });
    }
  }

  return result(method, params, slices, [], { order });
}

/** Demo point cloud for the "load example" path of the given-data workflow. */
export function loadDemoCloud(
  demo: Demo,
  nSamples = 1024,
  seed = 0,
): { x: Float64Array; y: Float64Array; n: number } {
  const port = sample(demo.problem, nSamples, {
    calcSecondOrder: false,
    seed,
    verbose: false,
  });
  const x = port.samples;
  const y = demo.evaluate({ samples: x, nParams: port.nParams });
  return { x, y, n: x.length / port.nParams };
}

// ---------------------------------------------------------------------------
// Given-data uploads (workflow B)
// ---------------------------------------------------------------------------

/**
 * Parse an X CSV for the given-data path: one header row naming the
 * parameters (must match the problem's names, in order) followed by N rows
 * of D values. Returns the point cloud row-major, (N, D).
 */
export function parseXGiven(parsed: ParsedCsv, problem: ProblemSpec): Float64Array {
  const D = problem.names.length;
  if (parsed.rows.length === 0) {
    throw new Error("the X CSV has no data rows");
  }
  const header = parsed.headers.map((h) => h.trim());
  if (header.length !== D || header.some((h, j) => h !== problem.names[j])) {
    throw new Error(
      `the X CSV header must be exactly: ${problem.names.join(", ")}`,
    );
  }
  for (let i = 0; i < parsed.rows.length; i++) {
    if (parsed.rows[i].length !== D) {
      throw new Error(`line ${i + 2}: expected ${D} columns, found ${parsed.rows[i].length}`);
    }
  }
  const flat = new Float64Array(parsed.rows.length * D);
  for (let i = 0; i < parsed.rows.length; i++) {
    for (let j = 0; j < D; j++) flat[i * D + j] = parsed.rows[i][j];
  }
  return flat;
}

// ---------------------------------------------------------------------------
// Results + session serialization
// ---------------------------------------------------------------------------

/** Serialize a result to CSV: one row per (output, time, parameter) slice. */
export function resultToCsv(result: AnalysisResult): string {
  const indexCols = result.slices[0]?.columns.map((c) => c.label) ?? [];
  const lines = [`output,time,parameter,${indexCols.join(",")}`];
  for (const s of result.slices) {
    for (let i = 0; i < result.parameters.length; i++) {
      lines.push(
        [
          s.output,
          String(s.time),
          result.parameters[i],
          ...s.columns.map((c) => String(c.values[i])),
        ].join(","),
      );
    }
  }
  return lines.join("\n") + "\n";
}

export interface SessionJson {
  app: "jaxgsa-web";
  version: 2;
  problem: ProblemSpec;
  /** One entry per generated design, keyed by method. */
  designs: Record<string, { csv: string; nRuns: number; summary: [string, string][] }>;
  y: {
    columns: Array<{ output: string; time: number; values: number[] }>;
    rowCount: number;
    label: string;
  } | null;
  /** Results with plain-number columns (JSON-serializable). */
  results: Array<{
    method: string;
    parameters: string[];
    notes: string[];
    settings: Record<string, unknown>;
    slices: Array<{
      output: string;
      time: number;
      columns: Array<{ key: string; label: string; values: number[] }>;
    }>;
  }>;
}

/**
 * Bundle the whole session — problem, generated designs (as CSV), the
 * labeled outputs, and every result — into one JSON document for download.
 */
export function buildSessionJson(
  problem: ProblemSpec,
  designs: Partial<Record<DesignMethod, GeneratedDesign>>,
  y: YData | null,
  results: AnalysisResult[],
): SessionJson {
  const designEntries: SessionJson["designs"] = {};
  for (const method of Object.keys(designs) as DesignMethod[]) {
    const gen = designs[method];
    if (gen) {
      designEntries[method] = {
        csv: buildDesignCsv(problem, gen),
        nRuns: gen.nRuns,
        summary: gen.summary,
      };
    }
  }
  return {
    app: "jaxgsa-web",
    version: 2,
    problem,
    designs: designEntries,
    y: y
      ? {
          columns: y.columns.map((c) => ({
            output: c.output,
            time: c.time,
            values: Array.from(c.values),
          })),
          rowCount: y.rowCount,
          label: y.label,
        }
      : null,
    results: results.map((r) => ({
      method: r.method,
      parameters: r.parameters,
      notes: r.notes,
      settings: r.settings,
      slices: r.slices.map((s) => ({
        output: s.output,
        time: s.time,
        columns: s.columns.map((c) => ({
          key: c.key,
          label: c.label,
          values: Array.from(c.values),
        })),
      })),
    })),
  };
}