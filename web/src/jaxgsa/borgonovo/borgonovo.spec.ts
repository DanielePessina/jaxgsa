import { describe, expect, it } from "vitest";
import type { ProblemSpec } from "@/jaxgsa/sampling";
import { loadBorgonovoGolden } from "@/goldens";
import { analyzeBorgonovo } from "./analyze";

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

describe("borgonovo port vs golden fixture", () => {
  it("matches the pinned Ishigami delta and S1 values", () => {
    const golden = loadBorgonovoGolden();
    const x = Float64Array.from(golden.x.flat());
    const y = Float64Array.from(golden.y);

    const { delta, S1 } = analyzeBorgonovo(ishigami, x, y);

    expect(delta.length).toBe(3);
    expect(S1.length).toBe(3);

    approx(delta, golden.expected.delta, golden.tolerance);
    approx(S1, golden.expected.S1, golden.tolerance);
  });

  it("returns exactly zero for a constant input column", () => {
    const N = 64;
    const x = new Float64Array(N * 2);

    for (let i = 0; i < N; i++) {
      x[i * 2] = i / N; // varies
      x[i * 2 + 1] = 0.5; // constant
    }

    const y = new Float64Array(N);

    for (let i = 0; i < N; i++) y[i] = x[i * 2] + 1;

    const problem: ProblemSpec = {
      names: ["x1", "x2"],
      marginals: [
        { kind: "uniform", low: 0, high: 1 },
        { kind: "uniform", low: 0, high: 1 },
      ],
    };

    const { delta, S1 } = analyzeBorgonovo(problem, x, y);

    expect(delta[1]).toBe(0);
    expect(S1[1]).toBe(0);
    expect(delta[0]).toBeGreaterThan(0);
  });
});