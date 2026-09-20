import { beforeAll, describe, expect, it } from "vitest";
import { defaultDevice, init } from "@jax-js/jax";
import { loadSobolGolden } from "../../goldens";
import { analyzeSobol } from "./analyze";
import { sample, type ProblemSpec } from "./sample";

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

/** Ishigami over U(-pi, pi): f = sin(x1) + 7 sin^2(x2) + 0.1 x3^4 sin(x1). */
function ishigami(x: Float64Array, row: number, dim: number): number {
  const x1 = x[row * dim];
  const x2 = x[row * dim + 1];
  const x3 = x[row * dim + 2];

  return Math.sin(x1) + 7 * Math.sin(x2) ** 2 + 0.1 * x3 ** 4 * Math.sin(x1);
}

function evaluate(x: Float64Array, dim: number): Float64Array {
  const n = x.length / dim;
  const y = new Float64Array(n);

  for (let i = 0; i < n; i++) y[i] = ishigami(x, i, dim);

  return y;
}

const ISHIGAMI_PROBLEM: ProblemSpec = {
  names: ["x1", "x2", "x3"],
  marginals: [
    { kind: "uniform", low: -Math.PI, high: Math.PI },
    { kind: "uniform", low: -Math.PI, high: Math.PI },
    { kind: "uniform", low: -Math.PI, high: Math.PI },
  ],
};

// Analytical Ishigami indices (x3 has zero first-order and 0.2436 total).
const ISHIGAMI_S1 = [0.3139, 0.4424, 0.0];

const ISHIGAMI_ST = [0.5576, 0.4424, 0.2436];

const SANITY_TOL = 0.06;

// ---------------------------------------------------------------------------
// Golden fixture
// ---------------------------------------------------------------------------

describe("analyzeSobol vs golden fixture (saltelli-jansen, first/total, scalar)", () => {
  let golden: ReturnType<typeof loadSobolGolden>;
  let S1: Float64Array;
  let ST: Float64Array;

  beforeAll(async () => {
    await init();
    defaultDevice("wasm");
    golden = loadSobolGolden();
  });

  it("the golden identity-map premise holds: n == base_n * (D + 2)", () => {
    expect(golden.x.length).toBe(golden.config.baseN * (3 + 2));
    expect(golden.y.length).toBe(golden.x.length);
    expect(golden.config.calcSecondOrder).toBe(false);
  });

  it("matches golden S1 within tolerance (identity expandedToUnique)", () => {
    const n = golden.y.length;
    const expandedToUnique = Int32Array.from({ length: n }, (_, i) => i);

    const { S1: s1, ST: st } = analyzeSobol(golden.y, 3, {
      expandedToUnique,
      baseN: golden.config.baseN,
    });

    S1 = s1;
    ST = st;
    console.log("S1 =", Array.from(S1));
    console.log("ST =", Array.from(ST));
    console.log("golden S1 =", golden.expected.S1);
    console.log("golden ST =", golden.expected.ST);
    expect(allclose(S1, golden.expected.S1, golden.tolerance)).toBe(true);
  });

  it("matches golden ST within tolerance", () => {
    expect(allclose(ST, golden.expected.ST, golden.tolerance)).toBe(true);
  });

  it("returns shape-(3) arrays", () => {
    expect(S1.length).toBe(3);
    expect(ST.length).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// Expansion path (dedup correctness)
// ---------------------------------------------------------------------------

describe("analyzeSobol expansion path vs direct expanded path", () => {
  let design: ReturnType<typeof sample>;
  let yUnique: Float64Array;
  let yExpanded: Float64Array;

  beforeAll(async () => {
    await init();
    defaultDevice("wasm");
    // base_n=16, D=3, first/total, unscrambled: row 1 of the Sobol' sequence
    // is 0.5 in every dimension, so A_1 == B_1 and the design dedups.
    design = sample(ISHIGAMI_PROBLEM, 1000, {
      baseN: 16,
      calcSecondOrder: false,
      scramble: false,
      seed: 0,
      verbose: false,
    });
    yUnique = evaluate(design.samples, design.nParams);
    yExpanded = new Float64Array(design.nExpanded);

    for (let i = 0; i < design.nExpanded; i++) {
      yExpanded[i] = yUnique[design.expandedToUnique[i]];
    }
  });

  it("the design really has duplicate rows (n_expanded > n_unique)", () => {
    expect(design.nExpanded).toBe(16 * (3 + 2));
    expect(design.nExpanded).toBeGreaterThan(design.samples.length / design.nParams);
  });

  it("gather path (unique y + expandedToUnique) equals direct expanded path", () => {
    const direct = analyzeSobol(yExpanded, 3, { baseN: 16 });

    const gathered = analyzeSobol(yUnique, 3, {
      baseN: 16,
      expandedToUnique: design.expandedToUnique,
    });

    console.log("nExpanded =", design.nExpanded, "nUnique =", design.samples.length / design.nParams);
    console.log("direct   S1 =", Array.from(direct.S1));
    console.log("gathered S1 =", Array.from(gathered.S1));
    console.log("direct   ST =", Array.from(direct.ST));
    console.log("gathered ST =", Array.from(gathered.ST));
    expect(allclose(direct.S1, Array.from(gathered.S1), { atol: 1e-12, rtol: 0 })).toBe(true);
    expect(allclose(direct.ST, Array.from(gathered.ST), { atol: 1e-12, rtol: 0 })).toBe(true);
    expect(allclose(gathered.S1, Array.from(direct.S1), { atol: 1e-12, rtol: 0 })).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Statistical sanity (JS sampler end-to-end)
// ---------------------------------------------------------------------------

describe("analyzeSobol statistical sanity on Ishigami (base_n=512, scrambled)", () => {
  let S1: Float64Array;
  let ST: Float64Array;

  beforeAll(async () => {
    await init();
    defaultDevice("wasm");

    const design = sample(ISHIGAMI_PROBLEM, 1000, {
      baseN: 512,
      calcSecondOrder: false,
      scramble: true,
      seed: 1,
      verbose: false,
    });

    const yUnique = evaluate(design.samples, design.nParams);
    ({ S1, ST } = analyzeSobol(yUnique, 3, {
      expandedToUnique: design.expandedToUnique,
    }));
    console.log("sanity S1 =", Array.from(S1));
    console.log("sanity ST =", Array.from(ST));
  });

  it("S1 lands within ±0.06 of the analytical first-order indices", () => {
    for (let j = 0; j < 3; j++) {
      expect(Math.abs(S1[j] - ISHIGAMI_S1[j])).toBeLessThanOrEqual(SANITY_TOL);
    }
  });

  it("ST lands within ±0.06 of the analytical total-order indices", () => {
    for (let j = 0; j < 3; j++) {
      expect(Math.abs(ST[j] - ISHIGAMI_ST[j])).toBeLessThanOrEqual(SANITY_TOL);
    }
  });

  it("all indices are finite", () => {
    for (const v of S1) expect(Number.isFinite(v)).toBe(true);

    for (const v of ST) expect(Number.isFinite(v)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

describe("analyzeSobol validation", () => {
  beforeAll(async () => {
    await init();
    defaultDevice("wasm");
  });

  it("raises when n_expanded is not divisible by step = D + 2", () => {
    const y = new Float64Array(2561); // 2561 / 5 = 512.2
    expect(() => analyzeSobol(y, 3)).toThrow(/not a multiple of the Saltelli group size/);
  });

  it("raises when calcSecondOrder=true (S2 not ported)", () => {
    const y = new Float64Array(2560);
    expect(() => analyzeSobol(y, 3, { calcSecondOrder: true })).toThrow(
      /calcSecondOrder=true is not supported yet/,
    );
  });

  it("raises when an explicit baseN disagrees with the derived base_n", () => {
    const y = new Float64Array(2560); // derived base_n = 512
    expect(() => analyzeSobol(y, 3, { baseN: 256 })).toThrow(/does not match/);
  });
});