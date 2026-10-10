/**
 * Pure logic behind the design form in the data panel: defaults, inline
 * validation, the config sent to `generateDesign`, and the check that the
 * form still matches the design in use.
 */

import { EFAST_LOW_N } from "./analysisGuards";
import type { DesignConfig, DesignMethod, GeneratedDesign } from "./engine";

/** Every design field, kept as text so a cleared input stays cleared. */
export interface DesignForm {
  baseN: string;
  nTrajectories: string;
  numLevels: string;
  nSamples: string;
  /** Empty means "use the studio default" (see `efastDefaultPoints`). */
  nPerCurve: string;
  seed: string;
}

export const DEFAULT_DESIGN_FORM: DesignForm = {
  baseN: "128",
  nTrajectories: "20",
  numLevels: "4",
  nSamples: "128",
  nPerCurve: "",
  seed: "0",
};

/** eFAST interference factor used by the browser design. */
const EFAST_M = 4;

/** eFAST minimum points per curve: 4*M^2*max(D-1, 1) + 1. */
export function efastMinPoints(D: number): number {
  return 4 * EFAST_M * EFAST_M * Math.max(D - 1, 1) + 1;
}

/**
 * Points per curve used when the field is empty. Above the hard minimum,
 * because a curve near the minimum biases the indices without any error.
 */
export function efastDefaultPoints(D: number): number {
  return Math.max(efastMinPoints(D), EFAST_LOW_N);
}

const DESIGN_FORM_KEYS = ["baseN", "nTrajectories", "numLevels", "nSamples", "nPerCurve", "seed"] as const satisfies readonly (keyof DesignForm)[];

/** What storage may hold for the form. Any field can be wrong at run time. */
export type StoredForm = Partial<Record<keyof DesignForm, string>> | null;

function isText(value: string | undefined): value is string {
  return Object.prototype.toString.call(value) === "[object String]";
}

/**
 * Keep only stored fields that are strings; the rest fall back to the
 * defaults. A stale or hand-edited entry must not break the form.
 */
export function mergeStoredForm(stored: StoredForm): DesignForm {
  const form: DesignForm = { ...DEFAULT_DESIGN_FORM };

  if (stored === null) return form;

  for (const key of DESIGN_FORM_KEYS) {
    const value = stored[key];

    if (isText(value)) form[key] = value;
  }

  return form;
}

function isInt(text: string): boolean {
  return text.trim() !== "" && Number.isInteger(Number(text));
}

function isPowerOfTwo(n: number): boolean {
  return n >= 1 && (n & (n - 1)) === 0;
}

function nextPowerOfTwo(n: number): number {
  let p = 1;

  while (p < n) p *= 2;

  return p;
}

/** Field name to message, for every field of `method` that is not valid. */
export function designFormErrors(
  method: DesignMethod,
  form: DesignForm,
  D: number,
): Partial<Record<keyof DesignForm, string>> {
  const errors: Partial<Record<keyof DesignForm, string>> = {};

  if (!isInt(form.seed)) errors.seed = "Seed must be a whole number.";

  if (method === "sobol") {
    const n = Number(form.baseN);

    if (!isInt(form.baseN) || n < 16 || !isPowerOfTwo(n)) {
      errors.baseN = "Base N must be a power of 2, at least 16 (16, 32, 64, …).";
    }
  } else if (method === "morris") {
    if (!isInt(form.nTrajectories) || Number(form.nTrajectories) < 2) {
      errors.nTrajectories = "Use a whole number, at least 2.";
    }

    if (!isInt(form.numLevels) || Number(form.numLevels) < 2) {
      errors.numLevels = "Use a whole number, at least 2.";
    }
  } else if (method === "kucherenko") {
    if (!isInt(form.nSamples) || Number(form.nSamples) < 2) {
      errors.nSamples = "Use a whole number, at least 2.";
    }
  } else if (form.nPerCurve.trim() !== "") {
    const min = efastMinPoints(D);

    if (!isInt(form.nPerCurve) || Number(form.nPerCurve) < min) {
      errors.nPerCurve = `Use a whole number, at least ${min}.`;
    }
  }

  return errors;
}

/** The `generateDesign` config for `method`. Call only when the form has no errors. */
export function designConfig(method: DesignMethod, form: DesignForm, D: number): DesignConfig {
  const seed = Number(form.seed);

  if (method === "sobol") return { baseN: Number(form.baseN), calcSecondOrder: false, seed };

  if (method === "morris") {
    return {
      nTrajectories: Number(form.nTrajectories),
      numLevels: Number(form.numLevels),
      seed,
    };
  }

  if (method === "kucherenko") return { nSamples: Number(form.nSamples), seed };

  return { nPerCurve: Number(form.nPerCurve || efastDefaultPoints(D)), M: EFAST_M, seed };
}

/** Upper bound on model runs for the form, or null when the form is not valid. */
export function plannedRuns(method: DesignMethod, form: DesignForm, D: number): number | null {
  if (Object.keys(designFormErrors(method, form, D)).length > 0) return null;

  if (method === "sobol") return Number(form.baseN) * (D + 2);

  if (method === "morris") return Number(form.nTrajectories) * (D + 1);

  // Kucherenko rounds its sample count up to a power of 2.
  if (method === "kucherenko") return nextPowerOfTwo(Number(form.nSamples)) * (2 * D + 1);

  return Number(form.nPerCurve || efastDefaultPoints(D)) * D;
}

function summaryValue(design: GeneratedDesign, key: string): string | undefined {
  return design.summary.find(([k]) => k === key)?.[1];
}

/** Short text for the settings a generated design was built with. */
export function describeDesign(method: DesignMethod, design: GeneratedDesign): string {
  if (method === "sobol" || method === "kucherenko") {
    return `base N ${summaryValue(design, "base_n")}`;
  }

  if (method === "morris") {
    return (
      `${summaryValue(design, "trajectories")} trajectories, ` +
      `${summaryValue(design, "grid levels")} levels`
    );
  }

  return `${summaryValue(design, "n_per_curve")} points per curve`;
}

/** Short text for the same settings, read from the form. */
export function describeForm(method: DesignMethod, form: DesignForm, D: number): string {
  if (method === "sobol") return `base N ${form.baseN}`;

  if (method === "kucherenko") {
    return `base N ${nextPowerOfTwo(Math.max(2, Number(form.nSamples)))}`;
  }

  if (method === "morris") {
    return `${form.nTrajectories} trajectories, ${form.numLevels} levels`;
  }

  return `${form.nPerCurve || efastDefaultPoints(D)} points per curve`;
}

/**
 * A sentence when the form shows settings that differ from the design in use,
 * else null. The seed is not stored in a design, so it is not compared.
 */
export function designMismatch(
  method: DesignMethod,
  form: DesignForm,
  design: GeneratedDesign,
  D: number,
  active = true,
): string | null {
  if (Object.keys(designFormErrors(method, form, D)).length > 0) return null;

  const inUse = describeDesign(method, design);
  const shown = describeForm(method, form, D);

  if (inUse === shown) return null;

  const lead = active ? "Design in use" : "This design";

  return `${lead}: ${inUse}. The form shows ${shown}. Press "Regenerate design" to apply it.`;
}

/** Sentence about duplicate points dropped from the design, or null when none. */
export function duplicatesNote(design: GeneratedDesign): string | null {
  if (summaryValue(design, "duplicates removed") !== undefined) return null;

  const expanded = Number(summaryValue(design, "n_expanded"));
  const removed = expanded - design.nRuns;

  if (!Number.isFinite(removed) || removed <= 0) return null;

  return (
    `${removed} duplicate points were removed: ${expanded} planned, ` +
    `${design.nRuns} unique. Evaluate only the ${design.nRuns} rows in the CSV.`
  );
}
