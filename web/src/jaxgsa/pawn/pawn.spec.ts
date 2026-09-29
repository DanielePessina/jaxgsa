import { describe, expect, it } from "vitest";
import type { ProblemSpec } from "@/jaxgsa/sampling";
import { loadPawnGolden } from "@/goldens";
import { analyzePawn } from "./analyze";

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

describe("pawn port vs golden fixture", () => {
  it("matches the pinned Ishigami PAWN values", () => {
    const golden = loadPawnGolden();
    const x = Float64Array.from(golden.x.flat());
    const y = Float64Array.from(golden.y);

    const { pawn } = analyzePawn(ishigami, x, y);

    expect(pawn.length).toBe(3);
    approx(pawn, golden.expected.pawn, golden.tolerance);
  });

  it("returns ~0 for a parameter that does not affect the output", () => {
    const N = 256;
    const x = new Float64Array(N * 2);

    for (let i = 0; i < N; i++) {
      x[i * 2] = i / N;
      x[i * 2 + 1] = 0.5;
    }

    const y = new Float64Array(N);

    for (let i = 0; i < N; i++) y[i] = x[i * 2];

    const problem: ProblemSpec = {
      names: ["x1", "x2"],
      marginals: [
        { kind: "uniform", low: 0, high: 1 },
        { kind: "uniform", low: 0, high: 1 },
      ],
    };

    const { pawn } = analyzePawn(problem, x, y);

    expect(pawn[1]).toBeLessThan(0.1);
    expect(pawn[0]).toBeGreaterThan(0.3);
  });
});