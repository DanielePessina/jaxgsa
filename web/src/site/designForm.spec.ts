import { describe, expect, it } from "vitest";
import {
  DEFAULT_DESIGN_FORM,
  designConfig,
  designFormErrors,
  designMismatch,
  duplicatesNote,
  efastDefaultPoints,
  efastMinPoints,
  mergeStoredForm,
  plannedRuns,
} from "./designForm";
import type { GeneratedDesign } from "./engine";

const form = (over: Partial<typeof DEFAULT_DESIGN_FORM> = {}) => ({
  ...DEFAULT_DESIGN_FORM,
  ...over,
});

function fakeDesign(
  method: GeneratedDesign["method"],
  summary: [string, string][],
  nRuns = 10,
): GeneratedDesign {
  // SAFETY: the pure helpers under test read only method, summary and nRuns.
  return { method, summary, nRuns } as GeneratedDesign;
}

describe("designFormErrors", () => {
  it("requires a power of 2 for Sobol base N", () => {
    expect(designFormErrors("sobol", form({ baseN: "100" }), 3).baseN).toMatch(/power of 2/);
    expect(designFormErrors("sobol", form({ baseN: "8" }), 3).baseN).toBeDefined();
    expect(designFormErrors("sobol", form({ baseN: "" }), 3).baseN).toBeDefined();
    expect(designFormErrors("sobol", form({ baseN: "1024" }), 3)).toEqual({});
  });

  it("lets the eFAST points field stay empty", () => {
    expect(designFormErrors("efast", form({ nPerCurve: "" }), 3)).toEqual({});
    expect(designFormErrors("efast", form({ nPerCurve: "5" }), 3).nPerCurve).toMatch(/at least/);
    expect(designFormErrors("efast", form({ nPerCurve: String(efastMinPoints(3)) }), 3)).toEqual({});
  });

  it("checks the seed and Morris fields", () => {
    expect(designFormErrors("morris", form({ seed: "x" }), 3).seed).toBeDefined();
    expect(designFormErrors("morris", form({ nTrajectories: "1" }), 3).nTrajectories).toBeDefined();
    expect(designFormErrors("morris", form({ nLevels: "1.5" }), 3).nLevels).toBeDefined();
  });
});

describe("designConfig and plannedRuns", () => {
  it("falls back to the studio eFAST default, above the minimum", () => {
    const n = efastDefaultPoints(3);
    expect(n).toBeGreaterThanOrEqual(257);
    expect(n).toBeGreaterThanOrEqual(efastMinPoints(3));
    expect(designConfig("efast", form(), 3)).toMatchObject({ nPerCurve: n, M: 4 });
    expect(plannedRuns("efast", form(), 3)).toBe(n * 3);
  });

  it("reads Kucherenko samples as a minimum run budget", () => {
    // D=3: 7 blocks. ceil(100 / 7) = 15, next power of 2 is 16, so 16 * 7 = 112 runs.
    expect(plannedRuns("kucherenko", form({ nSamples: "100" }), 3)).toBe(16 * 7);
    // ceil(4096 / 7) = 586, next power of 2 is 1024.
    expect(plannedRuns("kucherenko", form({ nSamples: "4096" }), 3)).toBe(1024 * 7);
    // An exact fit stays: 16 * 7 = 112 gives base N 16.
    expect(plannedRuns("kucherenko", form({ nSamples: "112" }), 3)).toBe(112);
    expect(plannedRuns("kucherenko", form({ nSamples: "2" }), 3)).toBe(2 * 7);
  });

  it("returns null for an invalid form", () => {
    expect(plannedRuns("sobol", form({ baseN: "100" }), 3)).toBeNull();
  });
});

describe("designMismatch", () => {
  const sobol = fakeDesign("sobol", [["base_n", "128"]]);

  it("is silent when the form matches the design", () => {
    expect(designMismatch("sobol", form(), sobol, 3)).toBeNull();
  });

  it("says which design is in use when the form differs", () => {
    expect(designMismatch("sobol", form({ baseN: "1024" }), sobol, 3)).toMatch(
      /Design in use: base N 128\. The form shows base N 1024/,
    );
  });

  it("says 'This design' when the design is not the active X source", () => {
    expect(designMismatch("sobol", form({ baseN: "1024" }), sobol, 3, false)).toMatch(
      /^This design: base N 128\./,
    );
  });

  it("stays silent while the form is invalid (the field shows the error)", () => {
    expect(designMismatch("sobol", form({ baseN: "100" }), sobol, 3)).toBeNull();
  });

  it("compares Kucherenko after rounding", () => {
    const kuch = fakeDesign("kucherenko", [["base_n", "16"]]);
    expect(designMismatch("kucherenko", form({ nSamples: "100" }), kuch, 3)).toBeNull();
  });
});

describe("duplicatesNote", () => {
  it("explains removed duplicates for Morris", () => {
    const d = fakeDesign("morris", [["n_expanded", "80"]], 63);
    expect(duplicatesNote(d)).toMatch(/17 duplicate points were removed.*63 unique/);
  });

  it("stays silent when the summary already reports duplicates or there are none", () => {
    expect(duplicatesNote(fakeDesign("sobol", [["n_expanded", "80"], ["duplicates removed", "4"]], 76))).toBeNull();
    expect(duplicatesNote(fakeDesign("morris", [["n_expanded", "80"]], 80))).toBeNull();
  });
});

describe("mergeStoredForm", () => {
  it("keeps string fields and falls back for the rest", () => {
    // SAFETY: the test feeds wrong-typed stored fields on purpose.
    const stored = { baseN: "256", seed: 7, nSamples: null, nPerCurve: ["x"] } as never;
    const merged = mergeStoredForm(stored);
    expect(merged.baseN).toBe("256");
    expect(merged.seed).toBe(DEFAULT_DESIGN_FORM.seed);
    expect(merged.nSamples).toBe(DEFAULT_DESIGN_FORM.nSamples);
    expect(merged.nPerCurve).toBe(DEFAULT_DESIGN_FORM.nPerCurve);
  });

  it("returns the defaults for a missing or non-object entry", () => {
    expect(mergeStoredForm(null)).toEqual(DEFAULT_DESIGN_FORM);
    // SAFETY: the test feeds a wrong-typed stored value on purpose.
    expect(mergeStoredForm(5 as never)).toEqual(DEFAULT_DESIGN_FORM);
  });
});
