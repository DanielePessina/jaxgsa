import { describe, expect, it, vi } from "vitest";
import {
  barGeometry,
  chartDomain,
  chartRowsForSlice,
  labelLayout,
  resolveCssVars,
  serializeForExport,
  shortenLabel,
  type ChartDataRow,
} from "./IndicesChart";
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


describe("barGeometry", () => {
  const yOf = (v: number) => 200 - v * 100; // zero line at 200

  it("returns null instead of NaN attributes for non-finite values", () => {
    expect(barGeometry(NaN, yOf, 200)).toBeNull();
    expect(barGeometry(Infinity, yOf, 200)).toBeNull();
    expect(barGeometry(-Infinity, yOf, 200)).toBeNull();
  });

  it("draws a positive bar upward from the zero line", () => {
    expect(barGeometry(0.5, yOf, 200)).toEqual({ y: 150, height: 50 });
  });

  it("draws a negative bar downward from the zero line", () => {
    expect(barGeometry(-0.25, yOf, 200)).toEqual({ y: 200, height: 25 });
  });

  it("keeps a one pixel bar for zero", () => {
    expect(barGeometry(0, yOf, 200)).toEqual({ y: 200, height: 1 });
  });
});

describe("labelLayout", () => {
  const width = 648;

  it("keeps a few short labels flat", () => {
    const layout = labelLayout(["x1", "x2", "x3"], width);

    expect(layout.rotate).toBe(false);
    expect(layout.stride).toBe(1);
  });

  it("rotates long names and cuts them", () => {
    const layout = labelLayout(["precipitation_rate", "soil_moisture_init", "albedo"], 200);

    expect(layout.rotate).toBe(true);
    expect(shortenLabel("precipitation_rate", layout.maxChars)).toHaveLength(layout.maxChars);
    expect(shortenLabel("precipitation_rate", layout.maxChars).endsWith("…")).toBe(true);
  });

  it("widens the left margin so the first rotated label stays inside", () => {
    const names = Array.from({ length: 30 }, (_, i) => `parameter_name_${i}`.padEnd(16, "x"));
    const layout = labelLayout(names, width);
    const step = width / names.length;
    const reach = 16 * 7.3 * Math.SQRT1_2 - step / 2;

    expect(layout.rotate).toBe(true);
    expect(layout.left).toBeGreaterThan(54);
    expect(layout.left).toBeGreaterThanOrEqual(reach);
  });

  it("keeps the default left margin for flat labels", () => {
    expect(labelLayout(["x1", "x2"], width).left).toBe(54);
  });

  it("thins the labels when there are many parameters", () => {
    const names = Array.from({ length: 120 }, (_, i) => `parameter_${i}`);
    const layout = labelLayout(names, width);

    expect(layout.rotate).toBe(true);
    expect(layout.stride).toBeGreaterThan(1);
  });
});

describe("resolveCssVars", () => {
  const values = new Map([
    ["--chart-grid", "#ddd"],
    ["--chart-missing", ""],
  ]);

  it("replaces every var() with the computed value", () => {
    const out = resolveCssVars(
      '<line stroke="var(--chart-grid)"/><rect fill="var(--chart-grid)"/>',
      (n) => values.get(n) ?? "",
    );

    expect(out).toBe('<line stroke="#ddd"/><rect fill="#ddd"/>');
  });

  it("falls back when the variable has no value", () => {
    expect(resolveCssVars('fill="var(--chart-missing)"', (n) => values.get(n) ?? "", "#000")).toBe(
      'fill="#000"',
    );
  });
});

describe("serializeForExport", () => {
  it("sets width and height on a clone, not on the live svg", () => {
    const attrs = new Map<string, string>();

    class FakeElement {
      setAttribute(k: string, v: string) {
        attrs.set(k, v);
      }
    }

    const clone = new FakeElement();
    const setAttribute = vi.fn();
    const live = { cloneNode: () => clone, setAttribute };

    vi.stubGlobal("Element", FakeElement);
    vi.stubGlobal("XMLSerializer", class {
      serializeToString() {
        return `<svg ${[...attrs].map(([k, v]) => `${k}="${v}"`).join(" ")}/>`;
      }
    });

    const out = serializeForExport(live);
    vi.unstubAllGlobals();

    expect(out).toBe('<svg width="720" height="400"/>');
    expect(setAttribute).not.toHaveBeenCalled();
  });
});
