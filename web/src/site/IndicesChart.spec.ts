import { describe, expect, it } from "vitest";
import { niceSteps, shapeChartData, type ChartDataRow } from "./IndicesChart";
import type { AnalysisResult } from "./engine";

const result: AnalysisResult = {
  method: "sobol",
  parameters: ["x1", "x2", "x3"],
  slices: [
    {
      output: "y",
      time: 0,
      columns: [
        { key: "S1", label: "S1", values: new Float64Array([0.32, 0.44, 0.0]) },
        { key: "ST", label: "ST", values: new Float64Array([0.56, 0.44, 0.24]) },
      ],
    },
    {
      output: "y",
      time: 1,
      columns: [
        { key: "S1", label: "S1", values: new Float64Array([0.1, 0.2, 0.3]) },
        { key: "ST", label: "ST", values: new Float64Array([0.4, 0.5, 0.6]) },
      ],
    },
  ],
  notes: [],
  settings: {},
};

describe("shapeChartData", () => {
  it("emits one row per parameter with one field per index column", () => {
    const rows: ChartDataRow[] = shapeChartData(result, 0);
    expect(rows).toEqual([
      { parameter: "x1", S1: 0.32, ST: 0.56 },
      { parameter: "x2", S1: 0.44, ST: 0.44 },
      { parameter: "x3", S1: 0, ST: 0.24 },
    ]);
  });

  it("selects the requested slice", () => {
    const rows = shapeChartData(result, 1);
    expect(rows[2]).toEqual({ parameter: "x3", S1: 0.3, ST: 0.6 });
  });

  it("throws for a missing slice index", () => {
    expect(() => shapeChartData(result, 9)).toThrow(/no slice 9/);
  });
});

describe("niceSteps", () => {
  it("picks a ~5-line major step and a 1/5 minor step", () => {
    expect(niceSteps(1)).toEqual({ major: 0.2, minor: 0.04 });
    expect(niceSteps(0.8)).toEqual({ major: 0.2, minor: 0.04 });
    expect(niceSteps(5)).toEqual({ major: 1, minor: 0.2 });
    expect(niceSteps(0.03)).toEqual({ major: 0.01, minor: 0.002 });
  });

  it("falls back for degenerate spans", () => {
    expect(niceSteps(0)).toEqual({ major: 0.2, minor: 0.04 });
    expect(niceSteps(Number.NaN)).toEqual({ major: 0.2, minor: 0.04 });
  });
});