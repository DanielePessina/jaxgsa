import { beforeAll, describe, expect, it } from "vitest";
import { defaultDevice, init } from "@jax-js/jax";
import { loadPceGolden } from "../../goldens";
import { analyzePce, fitPce } from "./analyze";
import {
  autoOrder,
  buildMultiIndex,
  comb,
  sobolFromCoefficients,
  type MultiIndex,
} from "./engine";
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

/** Flatten a (N, D) nested row-major matrix into a Float64Array. */
function flattenRows(x: number[][]): Float64Array {
  const n = x.length;
  const d = x[0].length;
  const out = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) out[i * d + j] = x[i][j];
  }
  return out;
}

function totalDegree(mi: MultiIndex, t: number): number {
  let s = 0;
  for (let d = 0; d < mi.D; d++) s += mi.flat[t * mi.D + d];
  return s;
}

const PCE_GOLDEN_PROBLEM: ProblemSpec = {
  names: ["x1", "x2", "x3"],
  marginals: [
    { kind: "uniform", low: -Math.PI, high: Math.PI },
    { kind: "uniform", low: -Math.PI, high: Math.PI },
    { kind: "uniform", low: -Math.PI, high: Math.PI },
  ],
};

// ---------------------------------------------------------------------------
// Golden fixture (shared 512-row Sobol A-block cloud, order 9)
// ---------------------------------------------------------------------------

describe("analyzePce vs golden fixture (Ishigami, order 9)", () => {
  let x: Float64Array;
  let y: Float64Array;
  let S1: Float64Array;
  let ST: Float64Array;
  let golden: ReturnType<typeof loadPceGolden>;

  beforeAll(async () => {
    await init();
    defaultDevice("wasm");
    golden = loadPceGolden();
    x = flattenRows(golden.x);
    y = Float64Array.from(golden.y);
    ({ S1, ST } = analyzePce(PCE_GOLDEN_PROBLEM, x, y, {
      order: golden.config.order,
      ridge: golden.config.ridge,
    }));
  });

  it("the golden cloud has the documented shape (512 rows, D=3)", () => {
    expect(golden.x.length).toBe(512);
    expect(golden.x[0].length).toBe(3);
    expect(golden.y.length).toBe(512);
    expect(golden.config.order).toBe(9);
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

  it("returns shape-(3) arrays", () => {
    expect(S1.length).toBe(3);
    expect(ST.length).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// Effective order (_auto_order)
// ---------------------------------------------------------------------------

describe("autoOrder (effective polynomial degree)", () => {
  it("keeps order 9 at D=3, N=512, fitRatio 0.5 (C(12,9)=220 <= 256)", () => {
    expect(comb(12, 9)).toBe(220);
    expect(comb(12, 9)).toBeLessThanOrEqual(256);
    expect(autoOrder(3, 512, 9, 0.5)).toBe(9);
  });

  it("reduces order 5 to 4 at D=3, N=100 (C(8,5)=56 > 50)", () => {
    expect(comb(8, 5)).toBe(56);
    expect(comb(7, 4)).toBe(35);
    expect(autoOrder(3, 100, 5, 0.5)).toBe(4);
  });

  it("never drops below order 1", () => {
    expect(autoOrder(3, 2, 9, 0.5)).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Multi-index construction
// ---------------------------------------------------------------------------

describe("buildMultiIndex", () => {
  it("builds n_terms = C(D+p, p) terms with row 0 all zeros", () => {
    const mi = buildMultiIndex(3, 9);
    expect(mi.nTerms).toBe(comb(12, 9));
    expect(mi.nTerms).toBe(220);
    expect(mi.flat.length).toBe(mi.nTerms * mi.D);
    for (let d = 0; d < mi.D; d++) expect(mi.flat[d]).toBe(0);
  });

  it("orders by total degree and lexicographically within each block", () => {
    const mi = buildMultiIndex(3, 2);
    expect(mi.nTerms).toBe(comb(5, 2));
    expect(mi.nTerms).toBe(10);
    // Python order for D=3, p=2: (0,0,0), then degree 1 lex, then degree 2 lex.
    const expected = [
      [0, 0, 0],
      [0, 0, 1],
      [0, 1, 0],
      [1, 0, 0],
      [0, 0, 2],
      [0, 1, 1],
      [0, 2, 0],
      [1, 0, 1],
      [1, 1, 0],
      [2, 0, 0],
    ];
    for (let t = 0; t < expected.length; t++) {
      for (let d = 0; d < 3; d++) expect(mi.flat[t * 3 + d]).toBe(expected[t][d]);
    }
  });

  it("keeps total degrees non-decreasing across all rows", () => {
    const mi = buildMultiIndex(3, 9);
    let prev = 0;
    for (let t = 0; t < mi.nTerms; t++) {
      const deg = totalDegree(mi, t);
      expect(deg).toBeGreaterThanOrEqual(prev);
      prev = deg;
    }
  });
});

// ---------------------------------------------------------------------------
// Fit sanity on the golden cloud
// ---------------------------------------------------------------------------

describe("analyzePce fit sanity (order 9 on the golden cloud)", () => {
  let coeffs: Float64Array;
  let S1: Float64Array;
  let ST: Float64Array;

  beforeAll(async () => {
    await init();
    defaultDevice("wasm");
    const golden = loadPceGolden();
    const x = flattenRows(golden.x);
    const y = Float64Array.from(golden.y);
    const fit = fitPce(PCE_GOLDEN_PROBLEM, x, y, {
      order: golden.config.order,
      ridge: golden.config.ridge,
    });
    expect(fit.coeffs.shape).toEqual([fit.mi.nTerms]);
    coeffs = fit.coeffs.ref.dataSync() as Float64Array; // ref'd: sobol consumes the array below
    ({ S1, ST } = sobolFromCoefficients(fit.coeffs, fit.mi));
  });

  it("fitted coefficients are finite", () => {
    expect(coeffs.length).toBe(220);
    for (const c of coeffs) expect(Number.isFinite(c)).toBe(true);
  });

  it("S1/ST are finite and within the loose [0, 1.2] band", () => {
    for (const v of S1) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1.2);
    }
    for (const v of ST) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1.2);
    }
  });

  it("ST >= S1 - 0.05 for every parameter", () => {
    for (let d = 0; d < 3; d++) {
      expect(ST[d]).toBeGreaterThanOrEqual(S1[d] - 0.05);
    }
  });
});