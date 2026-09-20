import { beforeAll, describe, expect, it } from "vitest";
import { defaultDevice, init, numpy as np } from "@jax-js/jax";
import { loadKucherenkoGolden } from "../goldens";
import { analyzeKucherenko } from "./kucherenko";

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

describe("kucherenko port vs golden fixture (Phase 0 spike)", () => {
  let golden: ReturnType<typeof loadKucherenkoGolden>;
  let S1: Float64Array;
  let ST: Float64Array;
  let variance: Float64Array;

  beforeAll(async () => {
    await init();
    defaultDevice("wasm");
    golden = loadKucherenkoGolden();

    const x = np.array(golden.x, { dtype: np.float64 });
    const y = np.array(golden.y, { dtype: np.float64 });
    const result = analyzeKucherenko(x, y);
    S1 = result.S1;
    ST = result.ST;
    variance = result.variance;

    console.log("S1       =", Array.from(S1));
    console.log("ST       =", Array.from(ST));
    console.log("variance =", Array.from(variance));
    console.log("golden S1 =", golden.expected.S1);
  });

  it("returns the golden shapes (S1/ST length 3, variance length 1)", () => {
    expect(S1.length).toBe(3);
    expect(ST.length).toBe(3);
    expect(variance.length).toBe(1);
  });

  it("matches golden S1 within tolerance", () => {
    expect(allclose(S1, golden.expected.S1, golden.tolerance)).toBe(true);
  });

  it("matches golden ST within tolerance", () => {
    expect(allclose(ST, golden.expected.ST, golden.tolerance)).toBe(true);
  });

  it("matches golden variance within tolerance", () => {
    expect(allclose(variance, golden.expected.variance, golden.tolerance)).toBe(true);
  });

  it("has positive variance (finite, well-conditioned denominator)", () => {
    expect(variance[0]).toBeGreaterThan(0);
    expect(Number.isFinite(variance[0])).toBe(true);
  });
});
