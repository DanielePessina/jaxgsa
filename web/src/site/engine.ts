/**
 * The site engine: jax-js device bootstrap, CSV/run_id exchange helpers, and
 * the thin typed wrappers that wire the React layer to the method ports.
 *
 * Ownership rule (see web/docs/PLAN-WASM.md): every port call consumes its input
 * arrays. The analyze ports build their own device copies from the host
 * arrays they receive, and `analyzeKucherenko` additionally consumes the
 * `np.Array` objects this module builds — so each workflow call creates
 * fresh arrays and never reuses a device buffer across calls.
 */

import { defaultDevice, init, numpy as np } from "@jax-js/jax";
import { analyzeBorgonovo, type BorgonovoIndices } from "@/jaxgsa/borgonovo/analyze";
import { analyzeEfast } from "@/jaxgsa/efast/analyze";
import { sampleEfast, type EfastDesign } from "@/jaxgsa/efast/sample";
import { analyzeKucherenko } from "@/jaxgsa/kucherenko";
import { sampleKucherenkoDesign, type KucherenkoDesign } from "@/jaxgsa/kucherenko/sample";
import { analyzeMorris, type MorrisMeasures } from "@/jaxgsa/morris/analyze";
import { sampleMorris, type MorrisDesign } from "@/jaxgsa/morris/sample";
import { analyzePawn, type PawnIndices } from "@/jaxgsa/pawn/analyze";
import { analyzePce, type PceDiagnostics, type PceIndices } from "@/jaxgsa/pce/analyze";
import { analyzeSobol, type SobolIndices } from "@/jaxgsa/sobol/analyze";
import { sample, type SobolDesign } from "@/jaxgsa/sobol/sample";
import { sobolSequence } from "@/jaxgsa/sobol/sampler";
import { transformSamples, type ProblemSpec } from "@/jaxgsa/sampling";
import { analyzeShapleyPce, type ShapleyPceResult } from "@/jaxgsa/shapley/analyze";
import { efastLowNWarning, pceFitWarnings, type SliceFit } from "./analysisGuards";
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
// Y exchange model (web/docs/PLAN-WEB-UI.md D1)
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

/** The `_t{time}` suffix. Time coordinates may be any finite decimal. */
const TIME_SUFFIX_RE = /^(.+)_t([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$/;

/**
 * Classify one output column name per the `_t{time}` exchange contract:
 * `<output>_t<number>` names that output at an arbitrary time coordinate;
 * anything else is `<name>` at time 0. A numeric `_t...` suffix is reserved —
 * an output literally named e.g. `foo_t2` must be written `foo_t2_t0`.
 */
export function classifyYColumn(name: string) {
  const m = TIME_SUFFIX_RE.exec(name);

  if (m) {
    const time = Number(m[2]);

    return { output: m[1], time: Object.is(time, -0) ? 0 : time };
  }

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
) {
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

  // Match the Python irregular-output contract: retain channel order, but
  // sort each channel's finite time coordinates rather than requiring the
  // upload header to already be chronological.
  const outputOrder = new Map<string, number>();

  for (const d of defs) {
    if (!outputOrder.has(d.output)) outputOrder.set(d.output, outputOrder.size);
  }

  defs.sort(
    (a, b) =>
      outputOrder.get(a.output)! - outputOrder.get(b.output)! || a.time - b.time,
  );

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
  const grids = new Map<string, number[]>();

  for (const c of y.columns) {
    const times = grids.get(c.output) ?? [];
    times.push(c.time);
    grids.set(c.output, times);
  }

  const entries = [...grids.values()];
  const first = entries[0] ?? [];

  const sharedGrid = entries.every(
    (times) => times.length === first.length && times.every((t, i) => t === first[i]),
  );

  const outputDesc = `${grids.size} output${grids.size === 1 ? "" : "s"}`;

  const timeDesc = sharedGrid
    ? `${first.length} time point${first.length === 1 ? "" : "s"}`
    : `${y.columns.length} slices · irregular time grids`;

  return `${outputDesc} · ${timeDesc} · ${y.rowCount} runs`;
}

/** Cap on the number of cells (rows x columns) any upload may hold. */
export const MAX_CELLS = 5_000_000;

/** A finite decimal: sign, digits, optional fraction, optional exponent. No hex, no "Infinity". */
const DECIMAL_RE = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

/** A number written with a decimal comma, e.g. "1,5" or "-0,25". */
const DECIMAL_COMMA_RE = /^[+-]?\d+,\d+$/;

/**
 * Parse one text cell as a finite decimal number. `Number()` alone would
 * accept "Infinity", "0x10" and "1e999", so the text is checked first.
 */
export function parseDecimal(
  raw: string,
): { value: number } | { reason: "not a number" | "not a finite number" } {
  const text = raw.trim();

  if (/^[+-]?inf(?:inity)?$/i.test(text)) return { reason: "not a finite number" };

  if (!DECIMAL_RE.test(text)) return { reason: "not a number" };

  const value = Number(text);

  if (!Number.isFinite(value)) return { reason: "not a finite number" };

  return { value };
}

const DELIMITERS = [",", ";", "\t"] as const;

/** Pick the delimiter that occurs most often (outside quotes) in the header line. */
function detectDelimiter(headerLine: string): string {
  const counts = new Map<string, number>(DELIMITERS.map((d) => [d, 0]));
  let inQuotes = false;

  for (const ch of headerLine) {
    if (ch === '"') inQuotes = !inQuotes;
    else if (!inQuotes && counts.has(ch)) counts.set(ch, (counts.get(ch) ?? 0) + 1);
  }

  let best: string = ",";
  let bestCount = 0;

  for (const d of DELIMITERS) {
    const c = counts.get(d) ?? 0;

    if (c > bestCount) {
      best = d;
      bestCount = c;
    }
  }

  return best;
}

/** Split one CSV line into trimmed fields, honouring double-quoted fields. */
function splitFields(line: string, delimiter: string, lineNo: number): string[] {
  if (!line.includes('"')) return line.split(delimiter).map((s) => s.trim());

  const fields: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];

    if (inQuotes) {
      if (ch !== '"') field += ch;
      else if (line[i + 1] === '"') {
        field += '"';
        i++;
      } else inQuotes = false;
    } else if (ch === '"') inQuotes = true;
    else if (ch === delimiter) {
      fields.push(field.trim());
      field = "";
    } else field += ch;
  }

  if (inQuotes) {
    throw new Error(
      `line ${lineNo}: a quoted field is not closed. ` +
        `Line breaks inside quoted fields are not supported`,
    );
  }

  fields.push(field.trim());

  return fields;
}

/**
 * Parse a numeric CSV: one header row, then finite decimal numbers. The
 * delimiter (comma, semicolon or tab) is detected from the header line.
 * Double-quoted fields, a UTF-8 BOM, CRLF line ends and blank lines are
 * tolerated. The size is checked from the line count before any row is built.
 * Throws on empty or non-numeric cells, naming the line and column.
 */
export function parseCsv(text: string, maxCells: number = MAX_CELLS): ParsedCsv {
  const body = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  const firstEnd = body.indexOf("\n");

  const headerLine = (firstEnd < 0 ? body : body.slice(0, firstEnd)).replace(/\r$/, "");

  if (body.trim() === "") return { headers: [], rows: [] };

  const delimiter = detectDelimiter(headerLine);
  const headers = splitFields(headerLine, delimiter, 1);

  // Upper bound on data rows: every remaining line, blank or not.
  let newlines = 0;

  for (let at = body.indexOf("\n"); at >= 0; at = body.indexOf("\n", at + 1)) newlines++;

  const rowBound = body.endsWith("\n") ? newlines - 1 : newlines;

  if (rowBound * headers.length > maxCells) {
    throw new Error(
      `file is too large: ${rowBound} rows x ${headers.length} ` +
        `columns exceeds the ${maxCells.toLocaleString()} cell limit`,
    );
  }

  const lines = body.split("\n");
  const rows: number[][] = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].replace(/\r$/, "");

    if (line.trim() === "") continue;

    const cells = splitFields(line, delimiter, i + 1);

    // A one-column file with decimal commas ("1,5") splits into two cells.
    if (headers.length === 1 && cells.length === 2 && DECIMAL_COMMA_RE.test(line.trim())) {
      throw new Error(
        `line ${i + 1}, column "${headers[0]}": "${line.trim()}" uses a decimal comma. ` +
          `Write decimals with a point (1.5)`,
      );
    }

    // Reject at the first row of the wrong width, before more rows are built.
    if (cells.length !== headers.length) {
      throw new Error(
        `line ${i + 1}: expected ${headers.length} columns, found ${cells.length}`,
      );
    }

    const row: number[] = [];

    for (let j = 0; j < cells.length; j++) {
      const where = `line ${i + 1}, column "${headers[j] ?? `#${j + 1}`}"`;

      if (cells[j] === "") throw new Error(`${where}: empty cell`);

      const parsed = parseDecimal(cells[j]);

      if ("reason" in parsed) {
        if (delimiter !== "," && DECIMAL_COMMA_RE.test(cells[j])) {
          throw new Error(
            `${where}: "${cells[j]}" uses a decimal comma. Write decimals with a ` +
              `point (1.5), or save the file with comma delimiters`,
          );
        }

        throw new Error(`${where}: "${cells[j]}" is ${parsed.reason}`);
      }

      row.push(parsed.value);
    }

    rows.push(row);
  }

  return { headers, rows };
}

/**
 * Why a parameter name cannot be written to (or read back from) a CSV
 * header, or null when it is fine. It is used by `buildDesignCsv` (the design
 * download) and by the problem table, which checks each parameter name.
 */
export function validateParameterName(name: string): string | null {
  if (name === "") return "A parameter name is empty.";

  if (name.trim() !== name) {
    return `Parameter name "${name}" has spaces at the start or end.`;
  }

  if (/[",;\t\r\n]/.test(name)) {
    return (
      `Parameter name "${name}" must not contain commas, semicolons, tabs, ` +
      `quotes or line breaks.`
    );
  }

  if (name === "run_id") {
    return 'Parameter name "run_id" is reserved for the run index.';
  }

  if (TIME_SUFFIX_RE.test(name)) {
    return (
      `Parameter name "${name}" must not end with _t and a number. ` +
      `That suffix marks a time point in the Y file.`
    );
  }

  return null;
}

/**
 * Serialize a design to CSV: a `run_id` column (0..n_runs-1, the canonical
 * row order) followed by one parameter column per problem marginal.
 * Throws when a parameter name would make the header unreadable.
 */
export function buildDesignCsv(
  problem: ProblemSpec,
  design: { samples: Float64Array },
): string {
  const seenNames = new Set<string>();

  for (const name of problem.names) {
    const bad = validateParameterName(name);

    if (bad) throw new Error(bad);

    if (seenNames.has(name)) throw new Error(`Parameter name "${name}" appears twice.`);
    seenNames.add(name);
  }

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

/**
 * Align Y columns onto an uploaded X by run_id. `yIds[i]` is the run_id of Y
 * row `i`; `xIds[k]` is the run_id of X row `k`. Both must hold exactly the
 * same set of ids. Returns the columns reordered into X row order.
 * Throws on a non-integer, duplicate, extra or missing id.
 */
export function alignColumnsToXRunIds(
  columns: YColumn[],
  yIds: number[],
  xIds: number[],
): YColumn[] {
  const position = new Map<number, number>();

  xIds.forEach((id, k) => position.set(id, k));

  const seen = new Set<number>();
  const target: number[] = [];

  for (let i = 0; i < yIds.length; i++) {
    const id = yIds[i];

    if (!Number.isInteger(id)) {
      throw new Error(`line ${i + 2}: invalid run_id ${id} (expected a whole number)`);
    }

    if (seen.has(id)) throw new Error(`line ${i + 2}: duplicate run_id ${id} in Y`);

    const at = position.get(id);

    if (at === undefined) {
      throw new Error(`line ${i + 2}: run_id ${id} is in Y but not in X`);
    }

    seen.add(id);
    target.push(at);
  }

  for (const id of xIds) {
    if (!seen.has(id)) {
      throw new Error(`run_id ${id} is in X but missing from Y: Y must cover every X row`);
    }
  }

  return columns.map((c) => {
    const out = new Float64Array(xIds.length);

    for (let i = 0; i < target.length; i++) out[target[i]] = c.values[i];

    return { output: c.output, time: c.time, values: out };
  });
}

// ---------------------------------------------------------------------------
// Design generation (workflow A)
// ---------------------------------------------------------------------------

export type DesignMethod = "sobol" | "morris" | "kucherenko" | "efast";

export type GivenDataMethod = "pce" | "shapley" | "borgonovo" | "pawn";

/** Where the X input comes from: a sampled design or an uploaded point cloud. */
export type XSource = { kind: "design"; method: DesignMethod } | { kind: "uploaded" };

export interface SobolConfig {
  baseN: number;
  calcSecondOrder: boolean;
  seed: number;
}

export interface MorrisConfig {
  nTrajectories: number;
  nLevels: number;
  seed: number;
}

export interface KucherenkoConfig {
  nSamples: number;
  seed: number;
}

export interface EfastConfig {
  nPerCurve: number;
  M: number;
  seed: number;
}

export type DesignConfig = SobolConfig | MorrisConfig | KucherenkoConfig | EfastConfig;

export type PortDesign = SobolDesign | MorrisDesign | KucherenkoDesign | EfastDesign;

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
    // SAFETY: this branch is selected by the method discriminator; callers pass SobolConfig here.
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
    // SAFETY: this branch is selected by the method discriminator; callers pass MorrisConfig here.
    const cfg = config as MorrisConfig;

    const port = sampleMorris(problem, cfg.nTrajectories, {
      nLevels: cfg.nLevels,
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
        ["n_levels", String(port.nLevels)],
        ["n_expanded", String(port.nExpanded)],
        ["n_runs", String(nRuns)],
      ],
      notes,
    };
  }

  if (method === "efast") {
    // SAFETY: this branch is selected by the method discriminator; callers pass EfastConfig here.
    const cfg = config as EfastConfig;
    const port = sampleEfast(problem, cfg.nPerCurve, { M: cfg.M, seed: cfg.seed });
    const nRuns = port.samples.length / D;

    return {
      method,
      problem,
      samples: port.samples,
      nRuns,
      nParams: D,
      port,
      summary: [
        ["n_per_curve", String(cfg.nPerCurve)],
        ["M", String(cfg.M)],
        ["n_runs", String(nRuns)],
      ],
      notes: [
        "eFAST has no bootstrap by design: a search curve is an ordered sweep, " +
          "so dropping a point changes what the Fourier transform computes.",
      ],
    };
  }

  // SAFETY: the remaining method branch is Kucherenko and therefore uses KucherenkoConfig.
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
  /** PCE fit quality for this slice (PCE and Shapley results only). */
  fit?: PceDiagnostics;
}

export interface AnalysisResult {
  method: string;
  parameters: string[];
  slices: ResultSlice[];
  notes: string[];
  /** The settings the analysis ran with (seed, order, base_n, ...) for repro. */
  settings: AnalysisSettings;
  /** Problems with the result the reader should see (poor fit, low N, ...). */
  warnings?: string[];
}

export type AnalysisSettings = Record<string, string | number | boolean | null>;

function result(
  method: string,
  parameters: string[],
  slices: ResultSlice[],
  notes: string[],
  settings: AnalysisSettings,
  warnings: string[] = [],
): AnalysisResult {
  return { method, parameters, slices, notes, settings, warnings };
}

/**
 * Run the analysis for a generated design. `y` carries one or more output
 * columns in run_id order (use `parseYColumns` + `alignColumnsByRunId` after
 * an upload). Every (output, time) slice is analyzed independently through
 * the scalar ports. Fresh device arrays are built per port call; nothing is
 * reused.
 */
export async function analyzeGenerated(gen: GeneratedDesign, y: YData): Promise<AnalysisResult> {
  const params = gen.problem.names;
  const slices: ResultSlice[] = [];

  for (const col of y.columns) {
    if (gen.method === "sobol") {
      // SAFETY: the generated design's method discriminator matches its Sobol port.
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
      // SAFETY: the generated design's method discriminator matches its Morris port.
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
    } else if (gen.method === "efast") {
      // SAFETY: the generated design's method discriminator matches its eFAST port.
      const design = gen.port as EfastDesign;

      const { S1, ST } = await analyzeEfast(design, col.values);
      slices.push({
        output: col.output,
        time: col.time,
        columns: [
          { key: "S1", label: "S1", values: S1 },
          { key: "ST", label: "ST", values: ST },
        ],
      });
    } else {
      // SAFETY: the remaining generated design is Kucherenko by the DesignMethod union.
      const design = gen.port as KucherenkoDesign;
      const nRuns = design.samples.length / design.nParams;

      // SAFETY: generated design samples are stored as host Float64Array values.
      const xNp = np
        .array(design.samples as Float64Array<ArrayBuffer>, { dtype: np.float64 })
        .reshape([nRuns, design.nParams]);

      // SAFETY: normalized Y columns own host Float64Array values.
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

  const warnings: string[] = [];

  if (gen.method === "efast") {
    // SAFETY: the generated design's method discriminator matches its eFAST port.
    const hint = efastLowNWarning((gen.port as EfastDesign).nPerCurve);

    if (hint) warnings.push(hint);
  }

  return result(gen.method, params, slices, gen.notes, Object.fromEntries(gen.summary), warnings);
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
  const usesOrder = method === "pce" || method === "shapley";

  if (usesOrder && (!Number.isInteger(order) || order < 1)) {
    throw new Error(`order must be a whole number >= 1, got ${order}`);
  }

  const params = problem.names;
  const xf = x instanceof Float64Array ? x : Float64Array.from(x);
  const N = xf.length / problem.names.length;

  if (!Number.isInteger(N)) {
    throw new Error(
      `x has ${xf.length} values, not a multiple of D=${problem.names.length}`,
    );
  }

  const slices: ResultSlice[] = [];
  const fits: SliceFit[] = [];
  let effectiveOrder = order;

  for (const col of y.columns) {
    if (col.values.length !== N) {
      throw new Error(
        `x has ${N} rows but output "${col.output}" (t${col.time}) has ${col.values.length} values`,
      );
    }

    if (method === "pce") {
      const fit: PceIndices = analyzePce(problem, xf, col.values, { order, diagnostics: true });
      effectiveOrder = fit.order;

      if (fit.diagnostics) {
        fits.push({ ...fit.diagnostics, label: yColumnLabel(col.output, col.time) });
      }

      slices.push({
        output: col.output,
        time: col.time,
        columns: [
          { key: "S1", label: "S1", values: fit.S1 },
          { key: "ST", label: "ST", values: fit.ST },
        ],
        fit: fit.diagnostics,
      });
    } else if (method === "shapley") {
      const fit: ShapleyPceResult = analyzeShapleyPce(problem, xf, col.values, {
        order,
        diagnostics: true,
      });

      effectiveOrder = fit.order;

      if (fit.diagnostics) {
        fits.push({ ...fit.diagnostics, label: yColumnLabel(col.output, col.time) });
      }

      slices.push({
        output: col.output,
        time: col.time,
        columns: [
          { key: "Sh", label: "Sh", values: fit.Sh },
          { key: "S1", label: "S1", values: fit.S1 },
          { key: "ST", label: "ST", values: fit.ST },
        ],
        fit: fit.diagnostics,
      });
    } else if (method === "borgonovo") {
      const { delta, S1 }: BorgonovoIndices = analyzeBorgonovo(problem, xf, col.values);
      slices.push({
        output: col.output,
        time: col.time,
        columns: [
          { key: "delta", label: "delta", values: delta },
          { key: "S1", label: "S1", values: S1 },
        ],
      });
    } else {
      const { pawn }: PawnIndices = analyzePawn(problem, xf, col.values);
      slices.push({
        output: col.output,
        time: col.time,
        columns: [{ key: "pawn", label: "PAWN", values: pawn }],
      });
    }
  }

  if (!usesOrder) return result(method, params, slices, [], {});

  // The effective order is what was fitted. autoOrder lowers the request when
  // the term count would not fit the sample budget.
  return result(
    method,
    params,
    slices,
    [],
    { order: effectiveOrder, order_requested: order },
    pceFitWarnings(fits, { requested: order, effective: effectiveOrder, nRows: N }),
  );
}

/** Demo point cloud for the "load example" path of the given-data workflow. */
export function loadDemoCloud(
  demo: Demo,
  nSamples = 1024,
  seed = 0,
) {
  const D = demo.problem.names.length;
  const unit = sobolSequence(D, nSamples, true, seed);
  const x = transformSamples(demo.problem, unit);
  const y = demo.evaluate({ samples: x, nParams: D });

  return { x, y, n: nSamples };
}

// ---------------------------------------------------------------------------
// Given-data uploads (workflow B)
// ---------------------------------------------------------------------------

/** An uploaded X: the point cloud, and the run_id of each row if the file had one. */
export interface XGiven {
  /** Row-major (N, D). */
  flat: Float64Array;
  /** One unique whole-number run_id per row, or null when the file has no run_id column. */
  runIds: number[] | null;
}

/**
 * Parse an X CSV for the given-data path: one header row naming the
 * parameters (must match the problem's names, in order) followed by N rows
 * of D values. A run_id column (from a design download) is allowed anywhere
 * and is returned separately, so Y can be aligned to it.
 */
export function parseXGiven(parsed: ParsedCsv, problem: ProblemSpec): XGiven {
  const D = problem.names.length;

  if (parsed.rows.length === 0) {
    throw new Error("the X CSV has no data rows");
  }

  const header = parsed.headers.map((h) => h.trim());

  const runIdIdx = header.indexOf("run_id");
  const keep = header.map((_, j) => j).filter((j) => j !== runIdIdx);

  if (keep.length !== D || keep.some((j, k) => header[j] !== problem.names[k])) {
    throw new Error(
      `the X CSV header must be exactly: ${problem.names.join(", ")} ` +
        `(a run_id column is allowed)`,
    );
  }

  for (let i = 0; i < parsed.rows.length; i++) {
    if (parsed.rows[i].length !== header.length) {
      throw new Error(
        `line ${i + 2}: expected ${header.length} columns, found ${parsed.rows[i].length}`,
      );
    }
  }

  const flat = new Float64Array(parsed.rows.length * D);

  for (let i = 0; i < parsed.rows.length; i++) {
    for (let k = 0; k < D; k++) flat[i * D + k] = parsed.rows[i][keep[k]];
  }

  let runIds: number[] | null = null;

  if (runIdIdx >= 0) {
    runIds = parsed.rows.map((r) => r[runIdIdx]);

    const seen = new Set<number>();

    runIds.forEach((id, i) => {
      if (!Number.isInteger(id)) {
        throw new Error(`line ${i + 2}: invalid run_id ${id} in X (expected a whole number)`);
      }

      if (seen.has(id)) throw new Error(`line ${i + 2}: duplicate run_id ${id} in X`);
      seen.add(id);
    });
  }

  return { flat, runIds };
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
  app: "jaxgsa-studio";
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
    settings: AnalysisSettings;
    warnings: string[];
    slices: Array<{
      output: string;
      time: number;
      columns: Array<{ key: string; label: string; values: number[] }>;
      fit?: PceDiagnostics;
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

  for (const method of ["sobol", "morris", "kucherenko", "efast"] as const) {
    if (!designs[method]) continue;
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
    app: "jaxgsa-studio",
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
      warnings: r.warnings ?? [],
      slices: r.slices.map((s) => ({
        output: s.output,
        time: s.time,
        fit: s.fit,
        columns: s.columns.map((c) => ({
          key: c.key,
          label: c.label,
          values: Array.from(c.values),
        })),
      })),
    })),
  };
}
