import { beforeAll, describe, expect, it } from "vitest";
import { defaultDevice, init } from "@jax-js/jax";
import { loadShapleyGolden } from "../../goldens";
import { analyzePce } from "../pce/analyze";
import { buildMultiIndex, multiIndexColumns } from "../pce/engine";
import { analyzeShapleyPce, shapleyFromPceCoefficients } from "./analyze";
import type { ProblemSpec } from "../sampling";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function allclose(
  actual: Float64Array,
  expected: number[],
  tol: { atol: number; rtol: number },
): boolean {
  if (actual.length !== expected.length) return false;

  for (let i = 0; i < actual.length; i++) {
    if (!(Math.abs(actual[i] - expected[i]) <= tol.atol + tol.rtol * Math.abs(expected[i]))) {
      return false;
    }
  }

  return true;
}

function flattenRows(x: number[][]): Float64Array {
  const n = x.length;
  const d = x[0].length;
  const out = new Float64Array(n * d);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) out[i * d + j] = x[i][j];
  }

  return out;
}

const SHAPLEY_GOLDEN_PROBLEM: ProblemSpec = {
  names: ["x1", "x2", "x3"],
  marginals: [
    { kind: "uniform", low: -Math.PI, high: Math.PI },
    { kind: "uniform", low: -Math.PI, high: Math.PI },
    { kind: "uniform", low: -Math.PI, high: Math.PI },
  ],
};

// ---------------------------------------------------------------------------
// Pure aggregation unit test (hand-computed D=2 case)
// ---------------------------------------------------------------------------

describe("shapleyFromPceCoefficients aggregation (D=2, order 2)", () => {
  it("matches the hand-computed Sh/S1/ST from the multi-index", () => {
    const mi = buildMultiIndex(2, 2);
    // Rows: (0,0),(0,1),(1,0),(0,2),(1,1),(2,0).
    const coeffs = Float64Array.from([10, 1, 2, 3, 4, 5]);
    // partial = [1,4,9,16,25], total 55:
    //   Sh = [37/55, 18/55], S1 = [29/55, 10/55], ST = [45/55, 26/55]
    const { Sh, S1, ST } = shapleyFromPceCoefficients(coeffs, multiIndexColumns(mi), 2);

    const close = (a: Float64Array, b: number[]) =>
      allclose(a, b, { atol: 1e-12, rtol: 0 });

    expect(close(Sh, [37 / 55, 18 / 55])).toBe(true);
    expect(close(S1, [29 / 55, 10 / 55])).toBe(true);
    expect(close(ST, [45 / 55, 26 / 55])).toBe(true);
    let shSum = 0;

    for (const v of Sh) shSum += v;
    expect(Math.abs(shSum - 1)).toBeLessThanOrEqual(1e-12);
  });
});

// ---------------------------------------------------------------------------
// Golden fixture (same 512-row cloud, order 9, backend pce)
// ---------------------------------------------------------------------------

describe("analyzeShapleyPce vs golden fixture (backend pce, order 9)", () => {
  let golden: ReturnType<typeof loadShapleyGolden>;
  let Sh: Float64Array;
  let S1: Float64Array;
  let ST: Float64Array;

  beforeAll(async () => {
    await init();
    defaultDevice("wasm");
    golden = loadShapleyGolden();
    const x = flattenRows(golden.x);
    const y = Float64Array.from(golden.y);
    ({ Sh, S1, ST } = analyzeShapleyPce(SHAPLEY_GOLDEN_PROBLEM, x, y, {
      order: golden.config.order,
    }));
  });

  it("the golden config uses the pce backend", () => {
    expect(golden.config.backend).toBe("pce");
    expect(golden.config.order).toBe(9);
  });

  it("matches golden Sh within tolerance", () => {
    console.log("Sh =", Array.from(Sh));
    console.log("golden Sh =", golden.expected.Sh);
    expect(allclose(Sh, golden.expected.Sh, golden.tolerance)).toBe(true);
  });

  it("matches golden S1 within tolerance", () => {
    console.log("S1 =", Array.from(S1));
    console.log("golden S1 =", golden.expected.S1);
    expect(allclose(S1, golden.expected.S1, golden.tolerance)).toBe(true);
  });

  it("matches golden ST within tolerance", () => {
    console.log("ST =", Array.from(ST));
    console.log("golden ST =", golden.expected.ST);
    expect(allclose(ST, golden.expected.ST, golden.tolerance)).toBe(true);
  });

  it("Sh sums to 1 within 1e-9", () => {
    let sum = 0;

    for (const v of Sh) sum += v;
    console.log("sum(Sh) =", sum);
    expect(Math.abs(sum - 1)).toBeLessThanOrEqual(1e-9);
  });

  it("returns shape-(3) arrays", () => {
    expect(Sh.length).toBe(3);
    expect(S1.length).toBe(3);
    expect(ST.length).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// Consistency: shapley S1/ST vs analyzePce S1/ST (same fit)
// ---------------------------------------------------------------------------

describe("analyzeShapleyPce consistency with analyzePce", () => {
  it("shapley S1/ST agree with analyzePce S1/ST within 1e-6", async () => {
    await init();
    defaultDevice("wasm");
    const golden = loadShapleyGolden();
    const x = flattenRows(golden.x);
    const y = Float64Array.from(golden.y);
    const opts = { order: golden.config.order };
    const pce = analyzePce(SHAPLEY_GOLDEN_PROBLEM, x, y, opts);
    const sh = analyzeShapleyPce(SHAPLEY_GOLDEN_PROBLEM, x, y, opts);
    console.log("pce   S1 =", Array.from(pce.S1));
    console.log("shap  S1 =", Array.from(sh.S1));
    console.log("pce   ST =", Array.from(pce.ST));
    console.log("shap  ST =", Array.from(sh.ST));
    expect(allclose(sh.S1, Array.from(pce.S1), { atol: 1e-6, rtol: 0 })).toBe(true);
    expect(allclose(sh.ST, Array.from(pce.ST), { atol: 1e-6, rtol: 0 })).toBe(true);
  });
});