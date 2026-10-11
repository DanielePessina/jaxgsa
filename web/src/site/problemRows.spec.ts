import { describe, expect, it } from "vitest";
import { DEMOS, demoById } from "./demos";
import {
  blankRow,
  clampStage,
  defaultRows,
  demoStillMatches,
  isProblemRows,
  problemKey,
  referenceDemo,
  rowsFromProblem,
  sameProblem,
  validateProblem,
} from "./problemRows";

describe("validateProblem", () => {
  it.each(DEMOS.map((d) => [d.id, d] as const))(
    "round-trips the %s demo through the table without change",
    (_id, demo) => {
      const { problem } = validateProblem(rowsFromProblem(demo.problem), "");

      expect(problem).not.toBeNull();
      expect(problemKey(problem!)).toBe(problemKey(demo.problem));
    },
  );

  it("starts from a valid default table", () => {
    expect(validateProblem(defaultRows(), "").issue).toBeNull();
  });

  it("reports a blank name and gives no problem", () => {
    const rows = defaultRows();
    rows[1] = { ...rows[1], name: " " };
    const v = validateProblem(rows, "");

    expect(v.problem).toBeNull();
    expect(v.issue).toBe("input 2: name is required");
  });

  it("flags only the rows with a repeated name", () => {
    const rows = [...defaultRows(), { ...blankRow(), name: "x1" }];
    const v = validateProblem(rows, "");

    expect(v.rowErrors).toEqual(["names must be unique", null, null, "names must be unique"]);
    expect(v.problem).toBeNull();
  });

  it("puts output names into the problem and changes its key", () => {
    const plain = validateProblem(defaultRows(), "").problem!;
    const named = validateProblem(defaultRows(), "a, b").problem!;

    expect(named.outputNames).toEqual(["a", "b"]);
    expect(sameProblem(plain, named)).toBe(false);
  });

  it("rejects duplicate output names", () => {
    const v = validateProblem(defaultRows(), "a,a");

    expect(v.problem).toBeNull();
    expect(v.issue).toContain("duplicate output name: a");
  });
});

describe("demo matching", () => {
  const ishigami = demoById("ishigami");

  it("keeps a demo only while the problem still equals it", () => {
    expect(demoStillMatches(ishigami, ishigami.problem)).toBe(ishigami);
    expect(demoStillMatches(ishigami, demoById("linear").problem)).toBeNull();
    expect(demoStillMatches(null, ishigami.problem)).toBeNull();
  });

  it("does not match an 8-input demo to a 3-input problem", () => {
    expect(demoStillMatches(demoById("sobol_g"), demoById("linear").problem)).toBeNull();
  });

  it("gives a reference only for Y made by the demo model", () => {
    const label = `demo model (${ishigami.label}) · 4 runs`;

    expect(referenceDemo(ishigami, ishigami.problem, label)).toBe(ishigami);
    expect(referenceDemo(ishigami, ishigami.problem, "my_outputs.csv")).toBeNull();
    expect(referenceDemo(ishigami, ishigami.problem, null)).toBeNull();
    expect(referenceDemo(null, ishigami.problem, label)).toBeNull();
    expect(referenceDemo(ishigami, demoById("linear").problem, label)).toBeNull();
  });

  it("gives no reference for user data on the default Linear problem", () => {
    const linear = demoById("linear");

    expect(referenceDemo(null, linear.problem, "uploaded.csv")).toBeNull();
  });
});

describe("isProblemRows", () => {
  it("accepts a real table and rejects empty or damaged ones", () => {
    expect(isProblemRows(defaultRows())).toBe(true);
    expect(isProblemRows([])).toBe(false);
    expect(isProblemRows(null)).toBe(false);
    // SAFETY: deliberately damaged row, to prove the guard rejects it.
    const damaged = [{ ...blankRow(), dist: "weird" as "uniform" }];

    expect(isProblemRows(damaged)).toBe(false);
  });
});

describe("clampStage", () => {
  const locked = [false, true, true, true];

  it("falls back to the furthest unlocked stage after a restore", () => {
    expect(clampStage(3, locked)).toBe(0);
    expect(clampStage(2, [false, false, true, true])).toBe(1);
  });

  it("keeps an unlocked stage", () => {
    expect(clampStage(2, [false, false, false, true])).toBe(2);
  });

  it("handles a missing stage", () => {
    expect(clampStage(-1, locked)).toBe(0);
    expect(clampStage(99, [false, false, false, false])).toBe(3);
  });
});

describe("isProblemRows field checks", () => {
  it("accepts the default rows", () => {
    expect(isProblemRows(defaultRows())).toBe(true);
  });

  it("rejects a row with a non-string field", () => {
    const base = JSON.stringify(blankRow());
    const parse = (patch: string) => JSON.parse(`[${base.slice(0, -1)},${patch}}]`);

    expect(isProblemRows(parse('"low":3'))).toBe(false);
    expect(isProblemRows(parse('"name":null'))).toBe(false);
    expect(isProblemRows(JSON.parse("[null]"))).toBe(false);
    expect(isProblemRows(JSON.parse("[1]"))).toBe(false);
  });
});

describe("parameter name validation in the table", () => {
  const withName = (name: string) => validateProblem([{ ...blankRow(), name }], "");

  it.each([
    ["a,b"],
    ['a"b'],
    ["a;b"],
    ["a\tb"],
    ["a\nb"],
    ["run_id"],
    ["x_t3"],
    [" x"],
    ["x "],
  ])("flags %j inline and blocks the problem", (name) => {
    const v = withName(name);

    expect(v.rowErrors[0]).not.toBeNull();
    expect(v.problem).toBeNull();
  });

  it("keeps the plain message for an empty name", () => {
    expect(withName("").rowErrors[0]).toBe("name is required");
  });

  it("accepts an ordinary name", () => {
    expect(withName("x1").problem).not.toBeNull();
  });
});
