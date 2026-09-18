import { beforeAll, describe, expect, it } from "vitest";
import { defaultDevice, init } from "@jax-js/jax";
import { ishigami } from "@/jaxgsa/benchmarks";
import { sampleKucherenkoDesign } from "@/jaxgsa/kucherenko/sample";
import {
  ISHIGAMI_PROBLEM,
  alignColumnsByRunId,
  analyzeGenerated,
  buildDesignCsv,
  buildSessionJson,
  classifyYColumn,
  describeY,
  generateDesign,
  parseCsv,
  parseXGiven,
  parseYColumns,
  resultToCsv,
  type AnalysisResult,
  type YData,
} from "./engine";

describe("parseCsv", () => {
  it("parses a header row and numeric rows", () => {
    const { headers, rows } = parseCsv("run_id,x1,x2\n0,1.5,-2.25\n1,3,4\n");
    expect(headers).toEqual(["run_id", "x1", "x2"]);
    expect(rows).toEqual([
      [0, 1.5, -2.25],
      [1, 3, 4],
    ]);
  });

  it("is tolerant of CRLF, trailing newlines and blank lines", () => {
    const { rows } = parseCsv("a,b\r\n1,2\r\n\r\n3,4\r\n\n");
    expect(rows).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  it("returns empty headers/rows for empty input", () => {
    expect(parseCsv("")).toEqual({ headers: [], rows: [] });
    expect(parseCsv("\n\n")).toEqual({ headers: [], rows: [] });
  });

  it("throws a clear error on non-numeric cells", () => {
    expect(() => parseCsv("a\n1\nnope\n")).toThrow(/not a number/);
  });

  it("throws on empty cells", () => {
    expect(() => parseCsv("a,b\n1,\n")).toThrow(/empty cell/);
  });
});

describe("buildDesignCsv", () => {
  it("emits run_id plus one parameter column per marginal", () => {
    const design = { samples: new Float64Array([1, 2, 3, 4, 5, 6]) };
    const csv = buildDesignCsv(ISHIGAMI_PROBLEM, design);
    const lines = csv.trim().split("\n");
    expect(lines[0]).toBe("run_id,x1,x2,x3");
    expect(lines).toHaveLength(3);
    expect(lines[1]).toBe("0,1,2,3");
    expect(lines[2]).toBe("1,4,5,6");
    expect(csv.endsWith("\n")).toBe(true);
  });
});

describe("classifyYColumn (the _t{time} contract)", () => {
  it("maps numeric time suffixes to arbitrary finite coordinates", () => {
    expect(classifyYColumn("y_t0")).toEqual({ output: "y", time: 0 });
    expect(classifyYColumn("pressure_t12")).toEqual({ output: "pressure", time: 12 });
    expect(classifyYColumn("pressure_t0.5")).toEqual({ output: "pressure", time: 0.5 });
    expect(classifyYColumn("pressure_t-1.25")).toEqual({ output: "pressure", time: -1.25 });
    expect(classifyYColumn("pressure_t2e-3")).toEqual({ output: "pressure", time: 0.002 });
    expect(classifyYColumn("pressure_t-0")).toEqual({ output: "pressure", time: 0 });
  });

  it("treats anything else as time 0", () => {
    expect(classifyYColumn("y")).toEqual({ output: "y", time: 0 });
    expect(classifyYColumn("temp")).toEqual({ output: "temp", time: 0 });
  });

  it("escape hatch: foo_t2_t0 is the output foo_t2 at time 0", () => {
    expect(classifyYColumn("foo_t2_t0")).toEqual({ output: "foo_t2", time: 0 });
  });
});

describe("parseYColumns (Y schema classification)", () => {
  it("parses a scalar y column and extracts run_id", () => {
    const parsed = parseCsv("run_id,y\n0,1\n1,2\n");
    const { columns, runIds, rowCount } = parseYColumns(parsed, ISHIGAMI_PROBLEM);
    expect(rowCount).toBe(2);
    expect(runIds).toEqual([0, 1]);
    expect(columns).toHaveLength(1);
    expect(columns[0].output).toBe("y");
    expect(columns[0].time).toBe(0);
    expect(Array.from(columns[0].values)).toEqual([1, 2]);
  });

  it("parses _t{t} suffixes into output/time slices", () => {
    const parsed = parseCsv("run_id,y_t0,y_t1\n0,1,10\n1,2,20\n");
    const { columns } = parseYColumns(parsed, ISHIGAMI_PROBLEM);
    expect(columns.map((c) => [c.output, c.time])).toEqual([
      ["y", 0],
      ["y", 1],
    ]);
  });

  it("groups channels in first-seen order and sorts each irregular time grid", () => {
    const parsed = parseCsv(
      "run_id,temp_t3,pressure_t4,temp_t0.5,pressure_t0\n0,1,2,3,4\n",
    );
    const { columns } = parseYColumns(parsed, ISHIGAMI_PROBLEM);
    expect(columns.map((c) => [c.output, c.time])).toEqual([
      ["temp", 0.5],
      ["temp", 3],
      ["pressure", 0],
      ["pressure", 4],
    ]);
  });

  it("parses multiple outputs with and without timepoints", () => {
    const parsed = parseCsv("run_id,temp,pressure_t0,pressure_t1\n0,1,2,3\n");
    const { columns } = parseYColumns(parsed, ISHIGAMI_PROBLEM);
    expect(columns.map((c) => [c.output, c.time])).toEqual([
      ["temp", 0],
      ["pressure", 0],
      ["pressure", 1],
    ]);
  });

  it("ignores X columns so the design CSV can be re-uploaded with outputs appended", () => {
    const parsed = parseCsv("run_id,x1,x2,x3,y\n0,1,2,3,4\n1,5,6,7,8\n");
    const { columns } = parseYColumns(parsed, ISHIGAMI_PROBLEM);
    expect(columns.map((c) => c.output)).toEqual(["y"]);
  });

  it("rejects a file with no output columns", () => {
    const parsed = parseCsv("run_id,x1,x2,x3\n0,1,2,3\n");
    expect(() => parseYColumns(parsed, ISHIGAMI_PROBLEM)).toThrow(/no output columns/);
  });

  it("rejects duplicate (output, time) columns", () => {
    const parsed = parseCsv("y,y\n1,2\n");
    expect(() => parseYColumns(parsed, ISHIGAMI_PROBLEM)).toThrow(/duplicate output column/);
  });

  it("rejects ragged rows", () => {
    const parsed = parseCsv("run_id,y\n0,1\n1\n");
    expect(() => parseYColumns(parsed, ISHIGAMI_PROBLEM)).toThrow(/expected 2 columns/);
  });

  it("rejects a duplicate run_id column", () => {
    const parsed = parseCsv("run_id,run_id,y\n0,0,1\n");
    expect(() => parseYColumns(parsed, ISHIGAMI_PROBLEM)).toThrow(/duplicate run_id/);
  });
});

describe("describeY", () => {
  const values = new Float64Array([1, 2]);

  it("describes a shared regular grid", () => {
    expect(describeY({
      columns: [
        { output: "a", time: 0, values },
        { output: "a", time: 0.5, values },
        { output: "b", time: 0, values },
        { output: "b", time: 0.5, values },
      ],
      rowCount: 2,
      label: "test",
    })).toBe("2 outputs · 2 time points · 2 runs");
  });

  it("calls out irregular grids instead of inferring max(time) + 1", () => {
    expect(describeY({
      columns: [
        { output: "a", time: 0.5, values },
        { output: "a", time: 3, values },
        { output: "b", time: 4, values },
      ],
      rowCount: 2,
      label: "test",
    })).toBe("2 outputs · 3 slices · irregular time grids · 2 runs");
  });
});

describe("alignColumnsByRunId (the run_id join, slice-aware)", () => {
  const design = { samples: new Float64Array(9), nParams: 3 };
  const design2 = { samples: new Float64Array(2), nParams: 1 };

  it("round-trips a shuffled run_id column to design row order for every column", () => {
    const parsed = parseCsv("run_id,y_t0,y_t1\n1,3.3,33\n0,1.1,11\n2,5.5,55\n");
    const { columns, runIds } = parseYColumns(parsed, ISHIGAMI_PROBLEM);
    const aligned = alignColumnsByRunId(columns, runIds!, design);
    expect(Array.from(aligned[0].values)).toEqual([1.1, 3.3, 5.5]);
    expect(Array.from(aligned[1].values)).toEqual([11, 33, 55]);
  });

  it("rejects duplicate run_ids", () => {
    const parsed = parseCsv("run_id,y\n0,1\n0,2\n");
    const { columns, runIds } = parseYColumns(parsed, ISHIGAMI_PROBLEM);
    expect(() => alignColumnsByRunId(columns, runIds!, design2)).toThrow(
      /duplicate run_id 0/,
    );
  });

  it("rejects a wrong row count", () => {
    const parsed = parseCsv("run_id,y\n0,1\n");
    const { columns, runIds } = parseYColumns(parsed, ISHIGAMI_PROBLEM);
    expect(() => alignColumnsByRunId(columns, runIds!, design2)).toThrow(
      /design has 2 rows but the uploaded Y CSV has 1/,
    );
  });

  it("rejects a run_id column of the wrong length", () => {
    const parsed = parseCsv("run_id,y\n0,1\n1,2\n");
    const { columns } = parseYColumns(parsed, ISHIGAMI_PROBLEM);
    expect(() => alignColumnsByRunId(columns, [0], design2)).toThrow(
      /expected 2 ids, found 1/,
    );
  });

  it("rejects out-of-range and non-integer run_ids", () => {
    const parsed = parseCsv("run_id,y\n0,1\n1,2\n");
    const { columns } = parseYColumns(parsed, ISHIGAMI_PROBLEM);
    expect(() => alignColumnsByRunId(columns, [7, 0], design2)).toThrow(
      /invalid run_id 7/,
    );
    expect(() => alignColumnsByRunId(columns, [1.5, 0], design2)).toThrow(
      /invalid run_id 1.5/,
    );
  });

  it("rejects duplicate run_ids (a permutation hole implies a duplicate)", () => {
    const parsed = parseCsv("run_id,y\n0,1\n0,2\n");
    const { columns, runIds } = parseYColumns(parsed, ISHIGAMI_PROBLEM);
    expect(() => alignColumnsByRunId(columns, runIds!, design2)).toThrow(
      /duplicate run_id 0/,
    );
  });
});

describe("parseXGiven / parseYColumns (given-data path)", () => {
  const csv = parseCsv("x1,x2,x3\n1,2,3\n4,5,6\n");

  it("flattens an X CSV row-major when the header matches the problem", () => {
    const flat = parseXGiven(csv, ISHIGAMI_PROBLEM);
    expect(Array.from(flat)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("rejects a header that does not match the problem names", () => {
    expect(() =>
      parseXGiven(parseCsv("a,b,c\n1,2,3\n"), ISHIGAMI_PROBLEM),
    ).toThrow(/header must be exactly: x1, x2, x3/);
  });

  it("rejects ragged rows", () => {
    expect(() =>
      parseXGiven(parseCsv("x1,x2,x3\n1,2\n"), ISHIGAMI_PROBLEM),
    ).toThrow(/expected 3 columns, found 2/);
  });

  it("rejects an empty X CSV", () => {
    expect(() => parseXGiven(parseCsv("x1,x2,x3\n"), ISHIGAMI_PROBLEM)).toThrow(
      /no data rows/,
    );
  });

  it("parses multi-output Y without run_id positionally", () => {
    const { columns, runIds, rowCount } = parseYColumns(
      parseCsv("y_t0,pressure\n1.5,99\n2.5,88\n"),
      ISHIGAMI_PROBLEM,
    );
    expect(runIds).toBeNull();
    expect(rowCount).toBe(2);
    expect(columns.map((c) => [c.output, c.time])).toEqual([
      ["y", 0],
      ["pressure", 0],
    ]);
    expect(Array.from(columns[0].values)).toEqual([1.5, 2.5]);
  });
});

describe("resultToCsv", () => {
  const result: AnalysisResult = {
    method: "sobol",
    parameters: ["x1", "x2"],
    slices: [
      {
        output: "y",
        time: 0,
        columns: [
          { key: "S1", label: "S1", values: new Float64Array([0.5, 0.25]) },
          { key: "ST", label: "ST", values: new Float64Array([0.6, 0.25]) },
        ],
      },
      {
        output: "y",
        time: 1,
        columns: [
          { key: "S1", label: "S1", values: new Float64Array([0.4, 0.3]) },
          { key: "ST", label: "ST", values: new Float64Array([0.5, 0.3]) },
        ],
      },
    ],
    notes: [],
    settings: {},
  };

  it("emits a header row plus one row per (output, time, parameter)", () => {
    const csv = resultToCsv(result);
    const lines = csv.trim().split("\n");
    expect(lines[0]).toBe("output,time,parameter,S1,ST");
    expect(lines[1]).toBe("y,0,x1,0.5,0.6");
    expect(lines[2]).toBe("y,0,x2,0.25,0.25");
    expect(lines[3]).toBe("y,1,x1,0.4,0.5");
    expect(lines[4]).toBe("y,1,x2,0.3,0.3");
  });
});

describe("buildSessionJson (v2)", () => {
  it("bundles problem, designs, y and results", () => {
    const y: YData = {
      columns: [{ output: "y", time: 0, values: new Float64Array([7, 8]) }],
      rowCount: 2,
      label: "uploaded Y · 2 runs",
    };
    const session = buildSessionJson(ISHIGAMI_PROBLEM, {}, y, []);
    expect(session.app).toBe("jaxgsa-web");
    expect(session.version).toBe(2);
    expect(session.problem.names).toEqual(["x1", "x2", "x3"]);
    expect(session.y?.columns[0].values).toEqual([7, 8]);
    expect(session.y?.columns[0].output).toBe("y");
    expect(session.results).toEqual([]);
  });

  it("serializes each generated design as CSV", () => {
    const gen = {
      method: "sobol" as const,
      problem: ISHIGAMI_PROBLEM,
      samples: new Float64Array([1, 2, 3, 4, 5, 6]),
      nRuns: 2,
      nParams: 3,
      port: null as never,
      summary: [["base_n", "2"]] as [string, string][],
      notes: [],
    };
    const session = buildSessionJson(ISHIGAMI_PROBLEM, { sobol: gen }, null, []);
    expect(session.designs.sobol?.csv.split("\n")[0]).toBe("run_id,x1,x2,x3");
    expect(session.designs.sobol?.nRuns).toBe(2);
  });

  it("serializes slices with plain-number columns", () => {
    const result: AnalysisResult = {
      method: "pce",
      parameters: ["x1"],
      slices: [
        {
          output: "temp",
          time: 3,
          columns: [
            { key: "S1", label: "S1", values: new Float64Array([0.7]) },
          ],
        },
      ],
      notes: [],
      settings: { order: 3 },
    };
    const session = buildSessionJson(ISHIGAMI_PROBLEM, {}, null, [result]);
    expect(session.results[0].slices[0]).toEqual({
      output: "temp",
      time: 3,
      columns: [{ key: "S1", label: "S1", values: [0.7] }],
    });
    expect(session.results[0].settings).toEqual({ order: 3 });
  });
});

describe("analyzeGenerated with multi-output Y (slice loop)", () => {
  beforeAll(async () => {
    await init();
    defaultDevice("wasm");
  });

  it("computes one slice per (output, time) column", () => {
    const gen = generateDesign("sobol", ISHIGAMI_PROBLEM, {
      baseN: 64,
      calcSecondOrder: false,
      seed: 0,
    });
    const n = gen.nRuns;
    const y0 = new Float64Array(n);
    const y1 = new Float64Array(n);
    for (let r = 0; r < n; r++) {
      const row = Array.from(gen.samples.subarray(r * 3, r * 3 + 3));
      y0[r] = ishigami(row);
      y1[r] = 2 * ishigami(row);
    }
    const y: YData = {
      columns: [
        { output: "y", time: 0, values: y0 },
        { output: "y", time: 1, values: y1 },
      ],
      rowCount: n,
      label: "two slices",
    };
    const res = analyzeGenerated(gen, y);
    expect(res.slices).toHaveLength(2);
    expect(res.slices.map((s) => [s.output, s.time])).toEqual([
      ["y", 0],
      ["y", 1],
    ]);
    for (const s of res.slices) {
      expect(s.columns.map((c) => c.key)).toEqual(["S1", "ST"]);
      expect(s.columns[0].values.length).toBe(3);
    }
    // Sobol indices are invariant to scaling of Y: both slices agree.
    for (let j = 0; j < res.slices[0].columns[0].values.length; j++) {
      expect(res.slices[0].columns[0].values[j]).toBeCloseTo(
        res.slices[1].columns[0].values[j],
        6,
      );
    }
    expect(res.settings["base_n"]).toBe("64");
  });
});

describe("sampleKucherenkoDesign", () => {
  it("builds base_n * (2D + 1) rows in physical units with the block layout", () => {
    const design = sampleKucherenkoDesign(ISHIGAMI_PROBLEM, 100, 42);
    const D = 3;
    expect(design.baseN).toBe(128);
    expect(design.samples.length / D).toBe(128 * (2 * D + 1));
    for (let i = 0; i < design.samples.length; i++) {
      expect(design.samples[i]).toBeGreaterThanOrEqual(-Math.PI);
      expect(design.samples[i]).toBeLessThanOrEqual(Math.PI);
    }
  });

  it("keeps joint x_i in the first-order block and redraws the others (block-major)", () => {
    const design = sampleKucherenkoDesign(ISHIGAMI_PROBLEM, 8, 1);
    const D = 3;
    const baseN = design.baseN;
    const blocks = 2 * D + 1;
    expect(design.samples.length / D).toBe(baseN * blocks);
    for (let k = 0; k < baseN; k++) {
      const joint = design.samples.subarray(k * D, k * D + D);
      const first = design.samples.subarray((baseN + k) * D, (baseN + k) * D + D);
      expect(first[0]).toBe(joint[0]); // x1 kept from the joint row
      expect(first[1]).not.toBe(joint[1]); // x2 redrawn
      expect(first[2]).not.toBe(joint[2]); // x3 redrawn
    }
  });
});
