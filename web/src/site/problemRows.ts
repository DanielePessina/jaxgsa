/**
 * Pure helpers for the problem table: the editable row model, validation into
 * a `ProblemSpec`, problem equality, demo matching and stage clamping. The
 * app owns the rows, so these helpers live outside any component.
 */

import type { MarginalSpec, ProblemSpec } from "@/jaxgsa/sampling";
import { DEMOS, type Demo } from "./demos";
import { validateParameterName } from "./engine";

export type DistKind = "uniform" | "gaussian";

export interface ProblemRow {
  name: string;
  dist: DistKind;
  low: string;
  high: string;
  mean: string;
  variance: string;
  gLow: string;
  gHigh: string;
}

interface RowSpecResult {
  spec: MarginalSpec | null;
  error: string | null;
}

export function blankRow(): ProblemRow {
  return {
    name: "",
    dist: "uniform",
    low: "0",
    high: "1",
    mean: "",
    variance: "",
    gLow: "",
    gHigh: "",
  };
}

export function rowToSpec(row: ProblemRow): RowSpecResult {
  if (row.name.trim() === "") return { spec: null, error: "name is required" };
  const nameError = validateParameterName(row.name);

  if (nameError !== null) return { spec: null, error: nameError };

  if (row.dist === "uniform") {
    const low = Number(row.low);
    const high = Number(row.high);

    if (row.low.trim() === "" || row.high.trim() === "" || Number.isNaN(low) || Number.isNaN(high)) {
      return { spec: null, error: "low and high must be numbers" };
    }

    if (low >= high) return { spec: null, error: "low must be < high" };

    return { spec: { kind: "uniform", low, high }, error: null };
  }

  const mean = Number(row.mean);
  const variance = Number(row.variance);

  if (row.mean.trim() === "" || row.variance.trim() === "" || Number.isNaN(mean) || Number.isNaN(variance)) {
    return { spec: null, error: "mean and variance must be numbers" };
  }

  if (variance <= 0) return { spec: null, error: "variance must be > 0" };
  const low = row.gLow.trim() === "" ? undefined : Number(row.gLow);
  const high = row.gHigh.trim() === "" ? undefined : Number(row.gHigh);

  if (low !== undefined && Number.isNaN(low)) return { spec: null, error: "lower bound must be a number" };

  if (high !== undefined && Number.isNaN(high)) return { spec: null, error: "upper bound must be a number" };

  if (low !== undefined && high !== undefined && low >= high) {
    return { spec: null, error: "lower bound must be < upper bound" };
  }

  return { spec: { kind: "gaussian", mean, variance, low, high }, error: null };
}

export function rowsFromProblem(p: ProblemSpec): ProblemRow[] {
  return p.names.map((name, j) => {
    const m = p.marginals[j];

    if (m.kind === "uniform") {
      return {
        name,
        dist: "uniform" as const,
        low: String(m.low),
        high: String(m.high),
        mean: "",
        variance: "",
        gLow: "",
        gHigh: "",
      };
    }

    if (m.kind !== "gaussian") {
      throw new Error("categorical marginals are not ported yet");
    }

    return {
      name,
      dist: "gaussian" as const,
      low: "",
      high: "",
      mean: String(m.mean),
      variance: String(m.variance),
      gLow: m.low === undefined ? "" : String(m.low),
      gHigh: m.high === undefined ? "" : String(m.high),
    };
  });
}

/**
 * Comma-separated optional output labels, normalized to a trimmed list.
 * Empty input yields `null` (the package default: outputs are just `y`).
 */
export function outputNamesFromInput(raw: string): string[] | null {
  const names = raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s !== "");

  if (names.length === 0) return null;
  const dup = names.find((n, i) => names.indexOf(n) !== i);

  if (dup !== undefined) throw new Error(`duplicate output name: ${dup}`);

  return names;
}

/** The rows a fresh session starts with: the Linear demo problem. */
export function defaultRows(): ProblemRow[] {
  const linear = DEMOS.find((d) => d.id === "linear");

  if (!linear) throw new Error("the linear demo is missing");

  return rowsFromProblem(linear.problem);
}

const ROW_STRING_FIELDS = ["name", "low", "high", "mean", "variance", "gLow", "gHigh"] as const;

const isString = (v: ProblemRow[keyof ProblemRow] | null | undefined): boolean =>
  Object.prototype.toString.call(v) === "[object String]";

const isObject = (v: ProblemRow | null | undefined): v is ProblemRow =>
  Object.prototype.toString.call(v) === "[object Object]";

/**
 * Guard for rows read back from storage. The storage layer already drops
 * entries of an older version, so this only catches an empty or damaged list.
 */
export function isProblemRows(value: ProblemRow[] | null): value is ProblemRow[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      (row) =>
        isObject(row) &&
        (row.dist === "uniform" || row.dist === "gaussian") &&
        ROW_STRING_FIELDS.every((key) => isString(row[key])),
    )
  );
}

export interface ProblemValidation {
  /** The valid problem, or `null` while the table has an error. */
  problem: ProblemSpec | null;
  /** One message (or `null`) per row. */
  rowErrors: (string | null)[];
  outputNamesError: string | null;
  /** The first thing to fix, in plain words, or `null` when valid. */
  issue: string | null;
}

/** Turn the table rows and the output-names text into a problem plus errors. */
export function validateProblem(rows: ProblemRow[], outputNamesRaw: string): ProblemValidation {
  const specs = rows.map(rowToSpec);
  const names = rows.map((r) => r.name.trim());

  const rowErrors = specs.map((s, i) => {
    if (s.error !== null) return s.error;

    return names[i] !== "" && names.indexOf(names[i]) !== names.lastIndexOf(names[i])
      ? "names must be unique"
      : null;
  });

  let outputNames: string[] | null = null;
  let outputNamesError: string | null = null;

  try {
    outputNames = outputNamesFromInput(outputNamesRaw);
  } catch (err) {
    outputNamesError = err instanceof Error ? err.message : String(err);
  }

  const firstRow = rowErrors.findIndex((e) => e !== null);

  let issue: string | null = null;

  if (rows.length < 1) issue = "add at least one input";
  else if (firstRow >= 0) issue = `input ${firstRow + 1}: ${rowErrors[firstRow]}`;
  else if (outputNamesError !== null) issue = `output names: ${outputNamesError}`;

  if (issue !== null) return { problem: null, rowErrors, outputNamesError, issue };

  const marginals: MarginalSpec[] = [];

  for (const s of specs) {
    if (s.spec === null) throw new Error("validated problem has a missing marginal");

    marginals.push(s.spec);
  }

  const problem: ProblemSpec = { names, marginals };

  if (outputNames) problem.outputNames = outputNames;

  return { problem, rowErrors, outputNamesError, issue: null };
}

/** A stable string key: two problems are the same exactly when keys match. */
export function problemKey(problem: ProblemSpec): string {
  return JSON.stringify(problem);
}

export function sameProblem(a: ProblemSpec | null, b: ProblemSpec | null): boolean {
  if (a === null || b === null) return a === b;

  return problemKey(a) === problemKey(b);
}

/** Keep the loaded demo only while the problem still equals its problem. */
export function demoStillMatches(demo: Demo | null, problem: ProblemSpec | null): Demo | null {
  return demo !== null && sameProblem(demo.problem, problem) ? demo : null;
}

/** Prefix of the Y label that `scalarYData` receives from the demo model. */
export function demoYLabelPrefix(demo: Demo): string {
  return `demo model (${demo.label})`;
}

/**
 * The demo to compare results against. It is set only when Y came from the
 * demo model and the problem still matches that demo. Otherwise it is `null`,
 * so user data never gets the analytical reference line.
 */
export function referenceDemo(
  activeDemo: Demo | null,
  problem: ProblemSpec | null,
  yLabel: string | null,
): Demo | null {
  const demo = demoStillMatches(activeDemo, problem);

  if (demo === null || yLabel === null) return null;

  return yLabel.startsWith(demoYLabelPrefix(demo)) ? demo : null;
}

/**
 * Move a wanted stage back to the furthest unlocked one at or before it.
 * Stage 0 is never locked, so the result is always valid.
 */
export function clampStage(index: number, locked: boolean[]): number {
  let i = Number.isInteger(index) ? Math.min(Math.max(index, 0), locked.length - 1) : 0;

  while (i > 0 && locked[i]) i--;

  return i;
}
