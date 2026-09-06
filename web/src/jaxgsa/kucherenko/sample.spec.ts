/**
 * Kucherenko sampler layout tests: the block-major sharing structure must
 * match the Python design exactly (verified against the golden's stored
 * design), and the indices on a JS-sampled design must reach the analytical
 * Ishigami values.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { defaultDevice, init, numpy as np } from "@jax-js/jax";
import { ishigami } from "@/jaxgsa/benchmarks";
import { analyzeKucherenko } from "@/jaxgsa/kucherenko";
import { loadKucherenkoGolden } from "@/goldens";
import { sampleKucherenkoDesign } from "./sample";
import { ISHIGAMI_PROBLEM } from "@/site/engine";

/**
 * Verify the intended sharing structure of a block-major design:
 * first-order block i keeps joint x_i and redraws the others; total block i
 * keeps the joint others and redraws x_i.
 */
function sharingStructure(x: number[][], D: number, N: number) {
  const joint = (k: number, j: number) => x[k][j];
  const first = (i: number, k: number, j: number) => x[(1 + i) * N + k][j];
  const total = (i: number, k: number, j: number) => x[(1 + D + i) * N + k][j];
  let firstOk = true;
  let totalOk = true;
  let firstOthersRedrawn = true;
  let totalOthersRedrawn = true;
  for (let k = 0; k < Math.min(N, 16); k++) {
    for (let i = 0; i < D; i++) {
      if (first(i, k, i) !== joint(k, i)) firstOk = false;
      if (total(i, k, i) === joint(k, i)) totalOthersRedrawn = false;
      for (let j = 0; j < D; j++) {
        if (j !== i) {
          if (first(i, k, j) === joint(k, j)) firstOthersRedrawn = false;
          if (total(i, k, j) !== joint(k, j)) totalOk = false;
        }
      }
    }
  }
  return { firstOk, totalOk, firstOthersRedrawn, totalOthersRedrawn };
}

function toRows(samples: Float64Array, D: number): number[][] {
  const rows: number[][] = [];
  const nRuns = samples.length / D;
  for (let r = 0; r < nRuns; r++) {
    const row: number[] = [];
    for (let j = 0; j < D; j++) row.push(samples[r * D + j]);
    rows.push(row);
  }
  return rows;
}

describe("kucherenko sampler: block-major sharing structure", () => {
  it("the python golden design satisfies the intended sharing", () => {
    const golden = loadKucherenkoGolden();
    const D = ISHIGAMI_PROBLEM.names.length;
    const N = golden.x.length / (2 * D + 1);
    expect(Number.isInteger(N)).toBe(true);
    const s = sharingStructure(golden.x, D, N);
    expect(s.firstOk).toBe(true);
    expect(s.firstOthersRedrawn).toBe(true);
    expect(s.totalOk).toBe(true);
    expect(s.totalOthersRedrawn).toBe(true);
  });

  it("the TS design satisfies the same sharing", () => {
    const design = sampleKucherenkoDesign(ISHIGAMI_PROBLEM, 512, 42);
    const D = design.nParams;
    const s = sharingStructure(toRows(design.samples, D), D, design.baseN);
    expect(s.firstOk).toBe(true);
    expect(s.firstOthersRedrawn).toBe(true);
    expect(s.totalOk).toBe(true);
    expect(s.totalOthersRedrawn).toBe(true);
  });

  it("has baseN * (2D+1) rows with a power-of-two base", () => {
    const design = sampleKucherenkoDesign(ISHIGAMI_PROBLEM, 300, 1);
    expect(design.baseN).toBe(512);
    expect(design.samples.length / design.nParams).toBe(512 * 7);
  });

  it("is deterministic per seed and differs across seeds", () => {
    const a = sampleKucherenkoDesign(ISHIGAMI_PROBLEM, 64, 7);
    const b = sampleKucherenkoDesign(ISHIGAMI_PROBLEM, 64, 7);
    const c = sampleKucherenkoDesign(ISHIGAMI_PROBLEM, 64, 8);
    expect(Array.from(a.samples)).toEqual(Array.from(b.samples));
    expect(Array.from(a.samples)).not.toEqual(Array.from(c.samples));
  });
});

describe("kucherenko sampler: indices on a JS-sampled design", () => {
  beforeAll(async () => {
    await init();
    defaultDevice("wasm");
  });

  it("reaches the analytical Ishigami S1/ST", () => {
    const design = sampleKucherenkoDesign(ISHIGAMI_PROBLEM, 512, 3);
    const D = design.nParams;
    const nRuns = design.samples.length / D;
    const y = new Float64Array(nRuns);
    const row: number[] = new Array(D);
    for (let r = 0; r < nRuns; r++) {
      for (let j = 0; j < D; j++) row[j] = design.samples[r * D + j];
      y[r] = ishigami(row);
    }
    const xNp = np
      .array(design.samples as Float64Array<ArrayBuffer>, { dtype: np.float64 })
      .reshape([nRuns, D]);
    const yNp = np.array(y as Float64Array<ArrayBuffer>, { dtype: np.float64 });
    const { S1, ST } = analyzeKucherenko(xNp, yNp);

    const s1Exp = [0.314, 0.443, 0.0];
    const stExp = [0.558, 0.443, 0.244];
    for (let i = 0; i < 3; i++) {
      expect(Math.abs(S1[i] - s1Exp[i])).toBeLessThan(0.06);
      expect(Math.abs(ST[i] - stExp[i])).toBeLessThan(0.06);
    }
  });
});