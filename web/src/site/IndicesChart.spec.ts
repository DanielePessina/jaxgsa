import { describe, expect, it } from "vitest";
import { chartDomain, chartRowsForSlice, type ChartDataRow } from "./IndicesChart";
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

describe("chartRowsForSlice", () => {
  it("emits one row per parameter with one field per index column", () => {
    const rows: ChartDataRow[] = chartRowsForSlice(result, 0);
    expect(rows).toEqual([
      { parameter: "x1", S1: 0.32, ST: 0.56 },
      { parameter: "x2", S1: 0.44, ST: 0.44 },
      { parameter: "x3", S1: 0, ST: 0.24 },
    ]);
  });

  it("selects the requested slice", () => {
    const rows = chartRowsForSlice(result, 1);
    expect(rows[2]).toEqual({ parameter: "x3", S1: 0.3, ST: 0.6 });
  });

  it("throws for a missing slice index", () => {
    expect(() => chartRowsForSlice(result, 9)).toThrow(/no slice 9/);
  });
});

describe("chartDomain", () => {
  it("keeps negative estimates visible around a zero baseline", () => {
    const [min, max] = chartDomain([-0.1, 0.6]);
    expect(min).toBeLessThan(-0.1);
    expect(max).toBeGreaterThan(0.6);
  });

  it("handles constant and non-finite input", () => {
    expect(chartDomain([0])).toEqual([0, 1.08]);
    expect(chartDomain([Number.NaN, Number.POSITIVE_INFINITY])).toEqual([0, 1.08]);
  });
});
