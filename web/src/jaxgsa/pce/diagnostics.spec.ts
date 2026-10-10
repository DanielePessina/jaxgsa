import { beforeAll, describe, expect, it } from "vitest";
import { defaultDevice, init } from "@jax-js/jax";
import type { ProblemSpec } from "../sampling";
import { analyzePce, fitDiagnostics } from "./analyze";
import { analyzeShapleyPce } from "../shapley/analyze";

// Reference numbers come from `jaxgsa.pce.analyze` (float64) on this exact
// design: a Kronecker sequence, so the cloud needs no stored fixture.
//
//   A = [0.6180339887498949, 0.7548776662466927, 0.5698402909980532]
//   u[i, j] = (0.5 + (i + 1) * A[j]) % 1.0,  x = -pi + 2 pi u
//   y = sin(x1) + 7 sin(x2)^2 + 0.1 x3^4 sin(x1)   (Ishigami)
const A = [0.6180339887498949, 0.7548776662466927, 0.5698402909980532];

const PROBLEM: ProblemSpec = {
  names: ["x1", "x2", "x3"],
  marginals: [
    { kind: "uniform", low: -Math.PI, high: Math.PI },
    { kind: "uniform", low: -Math.PI, high: Math.PI },
    { kind: "uniform", low: -Math.PI, high: Math.PI },
  ],
};

function cloud(n: number) {
  const x = new Float64Array(n * 3);
  const y = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 3; j++) {
      x[i * 3 + j] = -Math.PI + 2 * Math.PI * ((0.5 + (i + 1) * A[j]) % 1);
    }

    const [x1, x2, x3] = [x[i * 3], x[i * 3 + 1], x[i * 3 + 2]];
    y[i] = Math.sin(x1) + 7 * Math.sin(x2) ** 2 + 0.1 * x3 ** 4 * Math.sin(x1);
  }

  return { x, y };
}

const CASES = [
  // n, order, fitRatio, effective order, explained_variance, loo_rmse, std(Y), S1
  { n: 200, order: 1, fitRatio: 0.5, eff: 1, ev: 0.1867227723210198, loo: 3.388397857241682, std: 3.670962792833174, s1x1: 0.9953625557174193 },
  { n: 200, order: 2, fitRatio: 0.5, eff: 2, ev: 0.25007307458892336, loo: 3.3663113588135447, std: 3.670962792833174, s1x1: 0.7158675863686902 },
  { n: 200, order: 3, fitRatio: 0.5, eff: 3, ev: 0.506873032350081, loo: 2.879326645774798, std: 3.670962792833174, s1x1: 0.5908732457986507 },
  // The request (8) does not fit 200 rows at fit_ratio 0.5: autoOrder lowers it to 6.
  { n: 200, order: 8, fitRatio: 0.5, eff: 6, ev: 0.9915594578671498, loo: 0.8778739430424995, std: 3.670962792833174, s1x1: 0.3175190324117311 },
  { n: 400, order: 8, fitRatio: 0.5, eff: 8, ev: 0.9998562765129839, loo: 0.16610926440965085, std: 3.6868271210262757, s1x1: 0.3124134009314542 },
  // Overfit: in-sample 0.98, but LOO RMSE is 41 times std(Y).
  { n: 64, order: 5, fitRatio: 0.9, eff: 5, ev: 0.9823563272526447, loo: 158.6496235135393, std: 3.8716315397281296, s1x1: 0.062401492619454914 },
];

describe("PCE fit diagnostics vs jaxgsa.pce.analyze", () => {
  beforeAll(async () => {
    await init();
    defaultDevice("wasm");
  });

  for (const c of CASES) {
    it(`N=${c.n}, order ${c.order}, fit_ratio ${c.fitRatio}`, () => {
      const { x, y } = cloud(c.n);

      const r = analyzePce(PROBLEM, x, y, {
        order: c.order,
        fitRatio: c.fitRatio,
        diagnostics: true,
      });

      expect(r.order).toBe(c.eff);
      expect(r.S1[0]).toBeCloseTo(c.s1x1, 7);
      expect(r.diagnostics).toBeDefined();
      expect(r.diagnostics!.explainedVariance).toBeCloseTo(c.ev, 8);
      expect(r.diagnostics!.looRmse / c.loo).toBeCloseTo(1, 6);
      expect(r.diagnostics!.stdY).toBeCloseTo(c.std, 9);
    });
  }

  it("Shapley reports the same fit", () => {
    const { x, y } = cloud(200);
    const r = analyzeShapleyPce(PROBLEM, x, y, { order: 3, diagnostics: true });

    expect(r.order).toBe(3);
    expect(r.diagnostics!.explainedVariance).toBeCloseTo(0.506873032350081, 8);
    expect(r.diagnostics!.looRmse / 2.879326645774798).toBeCloseTo(1, 6);
  });

  it("skips the diagnostics unless asked", () => {
    const { x, y } = cloud(200);

    expect(analyzePce(PROBLEM, x, y, { order: 3 }).diagnostics).toBeUndefined();
  });
});

describe("fitDiagnostics", () => {
  it("reports NaN explained variance for a constant output", () => {
    const d = fitDiagnostics(
      Float64Array.from([2, 2, 2, 2]),
      Float64Array.from([2, 2, 2, 2]),
      Float64Array.from([0.1, 0.1, 0.1, 0.1]),
    );

    expect(Number.isNaN(d.explainedVariance)).toBe(true);
  });

  it("clips a leverage of 1 so the LOO error stays finite", () => {
    const d = fitDiagnostics(
      Float64Array.from([0, 1, 2, 3]),
      Float64Array.from([0, 1, 2, 2.5]),
      Float64Array.from([0.2, 0.2, 0.2, 1]),
    );

    expect(Number.isFinite(d.looRmse)).toBe(true);
  });
});
