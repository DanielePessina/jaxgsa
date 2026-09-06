/**
 * Demo problems: the jaxgsa benchmark functions packaged as loadable
 * problems, each with its closed-form reference S1/ST so users can validate
 * a pipeline end to end without writing a model.
 */

import { gaussianLinear, ishigami, linear, sobolG } from "@/jaxgsa/benchmarks";
import type { ProblemSpec } from "@/jaxgsa/sampling";

export type DemoId = "ishigami" | "linear" | "gaussian_linear" | "sobol_g";

export interface Demo {
  id: DemoId;
  label: string;
  formula: string;
  problem: ProblemSpec;
  /** Closed-form Sobol' indices of the benchmark, if known. */
  reference: { S1: number[]; ST: number[] };
  evaluate(design: { samples: Float64Array; nParams: number }): Float64Array;
}

function evaluateDemo(
  fn: (row: number[]) => number,
): (design: { samples: Float64Array; nParams: number }) => Float64Array {
  return (design) => {
    const D = design.nParams;
    const nRuns = design.samples.length / D;
    const y = new Float64Array(nRuns);
    const row: number[] = new Array(D);
    for (let r = 0; r < nRuns; r++) {
      for (let j = 0; j < D; j++) row[j] = design.samples[r * D + j];
      y[r] = fn(row);
    }
    return y;
  };
}

const PI = Math.PI;

// Sobol' G analytical indices: V_j = 1/(3(1+a_j)^2), V = prod(1+V_j) - 1,
// S1_j = V_j/V, ST_j = 1 - V_{-j}/V (Saltelli, Sobol' 1995).
const G_A = [0, 1, 4.5, 9, 99, 99, 99, 99];
const G_VJ = G_A.map((a) => 1 / (3 * (1 + a) ** 2));
const G_ONE_PLUS = G_VJ.map((v) => 1 + v);
const G_V = G_ONE_PLUS.reduce((p, v) => p * v, 1) - 1;
const G_S1 = G_VJ.map((v) => v / G_V);
const G_ST = G_ONE_PLUS.map((_, j) => {
  const vMinus = G_ONE_PLUS.reduce((p, v, k) => (k === j ? p : p * v), 1) - 1;
  return 1 - vMinus / G_V;
});

const ADDITIVE_S1_ST = [1, 4, 9].map((c) => c / 14);

export const DEMOS: Demo[] = [
  {
    id: "ishigami",
    label: "Ishigami",
    formula: "f(x) = sin(x1) + 7 sin(x2)² + 0.1 x3⁴ sin(x1)",
    problem: {
      names: ["x1", "x2", "x3"],
      marginals: [
        { kind: "uniform", low: -PI, high: PI },
        { kind: "uniform", low: -PI, high: PI },
        { kind: "uniform", low: -PI, high: PI },
      ],
    },
    reference: {
      S1: [0.3139, 0.4424, 0.0],
      ST: [0.5576, 0.4424, 0.2436],
    },
    evaluate: evaluateDemo(ishigami),
  },
  {
    id: "linear",
    label: "Linear",
    formula: "f(x) = x1 + 2 x2 + 3 x3",
    problem: {
      names: ["x1", "x2", "x3"],
      marginals: [
        { kind: "uniform", low: 0, high: 1 },
        { kind: "uniform", low: 0, high: 1 },
        { kind: "uniform", low: 0, high: 1 },
      ],
    },
    reference: { S1: ADDITIVE_S1_ST, ST: ADDITIVE_S1_ST },
    evaluate: evaluateDemo(linear),
  },
  {
    id: "gaussian_linear",
    label: "Gaussian linear",
    formula: "f(x) = x1 + 2 x2 + 3 x3, x ~ N(0, 1)",
    problem: {
      names: ["x1", "x2", "x3"],
      marginals: [
        { kind: "gaussian", mean: 0, variance: 1 },
        { kind: "gaussian", mean: 0, variance: 1 },
        { kind: "gaussian", mean: 0, variance: 1 },
      ],
    },
    reference: { S1: ADDITIVE_S1_ST, ST: ADDITIVE_S1_ST },
    evaluate: evaluateDemo(gaussianLinear),
  },
  {
    id: "sobol_g",
    label: "Sobol' G",
    formula: "f(x) = ∏ (|4 xⱼ − 2| + aⱼ)/(1 + aⱼ), a = (0, 1, 4.5, 9, 99, 99, 99, 99)",
    problem: {
      names: ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"],
      marginals: Array.from({ length: 8 }, () => ({ kind: "uniform" as const, low: 0, high: 1 })),
    },
    reference: { S1: G_S1, ST: G_ST },
    evaluate: evaluateDemo(sobolG),
  },
];

export function demoById(id: DemoId): Demo {
  const demo = DEMOS.find((d) => d.id === id);
  if (!demo) throw new Error(`unknown demo problem: ${id}`);
  return demo;
}

/** The demo whose problem definition matches `problem`, if any. */
export function demoForProblem(problem: ProblemSpec | null): Demo | null {
  if (!problem) return null;
  const key = JSON.stringify(problem);
  return DEMOS.find((d) => JSON.stringify(d.problem) === key) ?? null;
}