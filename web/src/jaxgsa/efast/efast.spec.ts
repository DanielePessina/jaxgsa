import { describe, expect, it } from "vitest";
import { loadEfastGolden } from "@/goldens";
import { analyzeEfast } from "./analyze";
import { frequencyPlan, minNPerCurve, sampleEfast } from "./sample";
import type { ProblemSpec } from "@/jaxgsa/sampling";

const ishigami: ProblemSpec = {
  names: ["x1", "x2", "x3"],
  marginals: [
    { kind: "uniform", low: -Math.PI, high: Math.PI },
    { kind: "uniform", low: -Math.PI, high: Math.PI },
    { kind: "uniform", low: -Math.PI, high: Math.PI },
  ],
};

function approx(got: Float64Array, want: number[], { rtol, atol }: { rtol: number; atol: number }) {
  for (let i = 0; i < want.length; i++) {
    const err = Math.abs(got[i] - want[i]);

    expect(err).toBeLessThan(atol + rtol * Math.abs(want[i]));
  }
}

describe("efast port vs golden fixture", () => {
  it("matches the pinned Ishigami S1/ST values", async () => {
    const golden = loadEfastGolden();
    const y = Float64Array.from(golden.y);

    const design = {
      samples: Float64Array.from(golden.x.flat()),
      nParams: 3,
      nPerCurve: golden.config.n_per_curve,
      M: golden.config.M,
    };

    const { S1, ST } = await analyzeEfast(design, y);

    approx(S1, golden.expected.S1, golden.tolerance);
    approx(ST, golden.expected.ST, golden.tolerance);
  });

  it("builds a valid search-curve design with the expected row count", () => {
    const D = 3;
    const N = 256;
    const design = sampleEfast(ishigami, N, { M: 4, seed: 42 });

    expect(design.nParams).toBe(D);
    expect(design.nPerCurve).toBe(N);
    expect(design.samples.length).toBe(N * D * D);

    const plan = frequencyPlan(D, N, 4);
    expect(plan.omega0).toBeGreaterThan(0);
    expect(plan.omegaCompl).toHaveLength(D - 1);
  });

  it("enforces the n_per_curve minimum", () => {
    expect(() => sampleEfast(ishigami, 8, { M: 4 })).toThrow(/below the minimum/);
    expect(minNPerCurve(3, 4)).toBe(4 * 16 * 2 + 1);
  });
});