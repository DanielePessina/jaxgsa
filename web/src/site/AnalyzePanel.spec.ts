import { describe, expect, it } from "vitest";
import type { ProblemSpec } from "@/jaxgsa/sampling";
import { availability } from "./AnalyzePanel";
import { demoById } from "./demos";
import { analyzeCloud, generateDesign, initEngine, scalarYData } from "./engine";

const independent: ProblemSpec = {
  names: ["x1", "x2"],
  marginals: [
    { kind: "uniform", low: 0, high: 1 },
    { kind: "gaussian", mean: 0, variance: 1 },
  ],
};

describe("analysis method availability", () => {
  it("allows PCE and Shapley on a generated Sobol design", () => {
    const methods = availability({ kind: "design", method: "sobol" }, independent);

    expect(methods.sobol.available).toBe(true);
    expect(methods.morris.available).toBe(false);
    expect(methods.pce.available).toBe(true);
    expect(methods.shapley.available).toBe(true);
  });

  it("allows PCE and Shapley on a generated Morris design", () => {
    const methods = availability({ kind: "design", method: "morris" }, independent);

    expect(methods.sobol.available).toBe(false);
    expect(methods.morris.available).toBe(true);
    expect(methods.pce.available).toBe(true);
    expect(methods.shapley.available).toBe(true);
  });

  it("keeps dedicated estimators tied to their own designs", () => {
    const methods = availability({ kind: "uploaded" }, independent);

    expect(methods.sobol.available).toBe(false);
    expect(methods.morris.available).toBe(false);
    expect(methods.pce.available).toBe(true);
    expect(methods.shapley.available).toBe(true);
  });

  it("ties kucherenko and efast to their own designs", () => {
    const kucherenko = availability({ kind: "design", method: "kucherenko" }, independent);

    expect(kucherenko.kucherenko.available).toBe(true);
    expect(kucherenko.sobol.available).toBe(false);
    expect(kucherenko.morris.available).toBe(false);
    expect(kucherenko.efast.available).toBe(false);

    const efast = availability({ kind: "design", method: "efast" }, independent);

    expect(efast.efast.available).toBe(true);
    expect(efast.kucherenko.available).toBe(false);
  });

  it("allows borgonovo and pawn on any X source", () => {
    const methods = availability({ kind: "design", method: "sobol" }, independent);

    expect(methods.borgonovo.available).toBe(true);
    expect(methods.pawn.available).toBe(true);

    const uploaded = availability({ kind: "uploaded" }, independent);

    expect(uploaded.borgonovo.available).toBe(true);
    expect(uploaded.pawn.available).toBe(true);
  });

  it("explains when browser PCE cannot handle a truncated Gaussian", () => {
    const problem: ProblemSpec = {
      names: ["x1"],
      marginals: [{ kind: "gaussian", mean: 0, variance: 1, low: -2 }],
    };

    const methods = availability({ kind: "design", method: "morris" }, problem);

    expect(methods.morris.available).toBe(true);
    expect(methods.pce.available).toBe(false);
    expect(methods.shapley.available).toBe(false);
    expect(methods.pce.reason).toMatch(/truncated Gaussian/);
  });

  it("runs PCE on the generated Sobol demo inputs and outputs", async () => {
    await initEngine();

    const demo = demoById("linear");

    const design = generateDesign("sobol", demo.problem, {
      baseN: 64,
      calcSecondOrder: false,
      seed: 0,
    });

    const y = scalarYData(demo.evaluate(design), "demo");

    const result = analyzeCloud("pce", demo.problem, design.samples, y, 3);

    expect(result.slices[0].columns[0].values).toHaveLength(3);
    expect(result.slices[0].columns[0].values[0]).toBeCloseTo(1 / 14, 2);
  });
});
