import { describe, expect, it } from "vitest";
import { sampleKucherenkoDesign } from "@/jaxgsa/kucherenko/sample";
import {
  ISHIGAMI_PROBLEM,
  alignYByRunId,
  buildDesignCsv,
  buildSessionJson,
  parseCsv,
  parseXGiven,
  parseYGiven,
  resultToCsv,
  type AnalysisResult,
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

describe("alignYByRunId (the run_id join)", () => {
  const design = {
    samples: new Float64Array(9),
    nParams: 3,
  };
  const design2 = {
    samples: new Float64Array(2),
    nParams: 1,
  };

  it("round-trips a shuffled run_id column back to design row order", () => {
    const y = [
      [3.3], // row with run_id 1
      [1.1], // run_id 0
      [5.5], // run_id 2
    ];
    const aligned = alignYByRunId(y, [1, 0, 2], design);
    expect(Array.from(aligned)).toEqual([1.1, 3.3, 5.5]);
  });

  it("takes the first column when a row has extra columns", () => {
    const aligned = alignYByRunId([[9, 99], [8, 88]], [0, 1], {
      samples: new Float64Array(2),
      nParams: 1,
    });
    expect(Array.from(aligned)).toEqual([9, 8]);
  });

  it("rejects duplicate run_ids", () => {
    expect(() => alignYByRunId([[1], [2]], [0, 0], design2)).toThrow(
      /duplicate run_id 0/,
    );
  });

  it("rejects a wrong row count", () => {
    expect(() => alignYByRunId([[1]], [0], design2)).toThrow(
      /design has 2 rows but the uploaded Y CSV has 1/,
    );
  });

  it("rejects a run_id column of the wrong length", () => {
    expect(() => alignYByRunId([[1], [2]], [0], design2)).toThrow(
      /expected 2 ids, found 1/,
    );
  });

  it("rejects out-of-range and non-integer run_ids", () => {
    expect(() => alignYByRunId([[1], [2]], [0, 7], design2)).toThrow(
      /invalid run_id 7/,
    );
    expect(() => alignYByRunId([[1], [2]], [0, 1.5], design2)).toThrow(
      /invalid run_id 1.5/,
    );
  });

  it("rejects a missing run_id (permutation hole)", () => {
    expect(() => alignYByRunId([[1], [2]], [0, 0], design2)).toThrow(
      /duplicate/,
    );
  });
});

describe("parseXGiven / parseYGiven", () => {
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

  it("takes the first column of a Y CSV", () => {
    const y = parseYGiven(parseCsv("y,extra\n1.5,99\n2.5,88\n"));
    expect(Array.from(y)).toEqual([1.5, 2.5]);
  });
});

describe("resultToCsv", () => {
  const result: AnalysisResult = {
    method: "sobol",
    parameters: ["x1", "x2"],
    columns: [
      { key: "S1", label: "S1", values: new Float64Array([0.5, 0.25]) },
      { key: "ST", label: "ST", values: new Float64Array([0.6, 0.25]) },
    ],
    notes: [],
  };

  it("emits a header row plus one row per parameter", () => {
    const csv = resultToCsv(result);
    const lines = csv.trim().split("\n");
    expect(lines[0]).toBe("parameter,S1,ST");
    expect(lines[1]).toBe("x1,0.5,0.6");
    expect(lines[2]).toBe("x2,0.25,0.25");
  });
});

describe("buildSessionJson", () => {
  it("bundles problem, designs, y and results", () => {
    const y = { label: "uploaded Y · 2 runs", values: new Float64Array([7, 8]) };
    const session = buildSessionJson(
      ISHIGAMI_PROBLEM,
      {},
      y,
      [],
    );
    expect(session.app).toBe("jaxgsa-web");
    expect(session.problem.names).toEqual(["x1", "x2", "x3"]);
    expect(session.y?.values).toEqual([7, 8]);
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