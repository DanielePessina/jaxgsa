import { beforeAll, describe, expect, it } from "vitest";
import { defaultDevice, init } from "@jax-js/jax";
import { loadMorrisGolden } from "../../goldens";
import { ishigami } from "../benchmarks";
import { analyzeMorris } from "./analyze";
import { sampleMorris, type MorrisDesign } from "./sample";
import type { ProblemSpec } from "../sampling";

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

function evaluate(design: MorrisDesign): Float64Array {
  const D = design.nParams;
  const n = design.samples.length / D;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    y[i] = ishigami([
      design.samples[i * D],
      design.samples[i * D + 1],
      design.samples[i * D + 2],
    ]);
  }
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

// ---------------------------------------------------------------------------
// Golden fixture
// ---------------------------------------------------------------------------

describe("analyzeMorris vs golden fixture (trajectory, scalar)", () => {
  let golden: ReturnType<typeof loadMorrisGolden>;
  let design: MorrisDesign;
  let measures: ReturnType<typeof analyzeMorris>;

  beforeAll(async () => {
    await init();
    defaultDevice("wasm");
    golden = loadMorrisGolden();
    const d = golden.design;
    const D = 3;
    const xFlat = new Float64Array(golden.x.length * D);
    for (let r = 0; r < golden.x.length; r++) {
      for (let j = 0; j < D; j++) xFlat[r * D + j] = golden.x[r][j];
    }
    design = {
      samples: xFlat,
      expandedToUnique: Int32Array.from(d.expanded_to_unique),
      nExpanded: d.n_expanded,
      nTrajectories: d.n_trajectories,
      numLevels: d.num_levels,
      eeIdxAfter: Int32Array.from(d.ee_idx_after.flat()),
      eeIdxBefore: Int32Array.from(d.ee_idx_before.flat()),
      eeDelta: Float64Array.from(d.ee_delta.flat()),
      nParams: D,
    };
    measures = analyzeMorris(design, golden.y);
    console.log("port   mu       =", Array.from(measures.mu));
    console.log("port   mu_star  =", Array.from(measures.mu_star));
    console.log("port   sigma    =", Array.from(measures.sigma));
    console.log("golden mu       =", golden.expected.mu);
    console.log("golden mu_star  =", golden.expected.mu_star);
    console.log("golden sigma    =", golden.expected.sigma);
  });

  it("the golden design bookkeeping is consistent", () => {
    expect(golden.design.n_expanded).toBe(160);
    expect(golden.design.n_trajectories * (design.nParams + 1)).toBe(golden.design.n_expanded);
    expect(golden.design.expanded_to_unique.length).toBe(160);
    expect(golden.y.length).toBe(61);
    // ee_delta entries are ±p / (2 (p - 1)) = ±4 / 6 = ±0.666...
    for (const v of design.eeDelta) {
      expect(Math.abs(Math.abs(v) - 4 / 6)).toBeLessThan(1e-15);
    }
    // expanding the unique rows by expandedToUnique yields 160 expanded rows.
    const nUnique = golden.x.length;
    for (const u of design.expandedToUnique) {
      expect(u).toBeGreaterThanOrEqual(0);
      expect(u).toBeLessThan(nUnique);
    }
  });

  it("matches golden mu within tolerance", () => {
    expect(allclose(measures.mu, golden.expected.mu, golden.tolerance)).toBe(true);
  });

  it("matches golden mu_star within tolerance", () => {
    expect(allclose(measures.mu_star, golden.expected.mu_star, golden.tolerance)).toBe(true);
  });

  it("matches golden sigma within tolerance", () => {
    expect(allclose(measures.sigma, golden.expected.sigma, golden.tolerance)).toBe(true);
  });

  it("returns shape-(3) arrays", () => {
    expect(measures.mu.length).toBe(3);
    expect(measures.mu_star.length).toBe(3);
    expect(measures.sigma.length).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// Expansion path (gather correctness)
// ---------------------------------------------------------------------------

describe("analyzeMorris expansion path vs direct expanded computation", () => {
  let design: MorrisDesign;
  let yUnique: Float64Array;
  let directMu: Float64Array;
  let directMuStar: Float64Array;
  let directSigma: Float64Array;

  beforeAll(async () => {
    await init();
    defaultDevice("wasm");
    // 40 trajectories, num_levels=4, D=3: 160 expanded rows over a 4^3 grid,
    // so exact duplicates are guaranteed and the design must dedup.
    design = sampleMorris(ISHIGAMI_PROBLEM, 40, { numLevels: 4, seed: 1, verbose: false });
    yUnique = evaluate(design);

    // Manually expand y and compute the reductions in plain JS.
    const r = design.nTrajectories;
    const D = design.nParams;
    const yExpanded = new Float64Array(design.nExpanded);
    for (let i = 0; i < design.nExpanded; i++) yExpanded[i] = yUnique[design.expandedToUnique[i]];
    const sums = new Float64Array(D);
    const absSums = new Float64Array(D);
    for (let t = 0; t < r; t++) {
      for (let j = 0; j < D; j++) {
        const after = design.eeIdxAfter[t * D + j];
        const before = design.eeIdxBefore[t * D + j];
        const ee = (yExpanded[after] - yExpanded[before]) / design.eeDelta[t * D + j];
        sums[j] += ee;
        absSums[j] += Math.abs(ee);
      }
    }
    directMu = new Float64Array(D);
    directMuStar = new Float64Array(D);
    for (let j = 0; j < D; j++) {
      directMu[j] = sums[j] / r;
      directMuStar[j] = absSums[j] / r;
    }
    const sqSums = new Float64Array(D);
    for (let t = 0; t < r; t++) {
      for (let j = 0; j < D; j++) {
        const after = design.eeIdxAfter[t * D + j];
        const before = design.eeIdxBefore[t * D + j];
        const ee = (yExpanded[after] - yExpanded[before]) / design.eeDelta[t * D + j];
        sqSums[j] += (ee - directMu[j]) * (ee - directMu[j]);
      }
    }
    directSigma = new Float64Array(D);
    for (let j = 0; j < D; j++) directSigma[j] = Math.sqrt(sqSums[j] / (r - 1));
    console.log("nExpanded =", design.nExpanded, "nUnique =", design.samples.length / D);
    console.log("direct   mu =", Array.from(directMu));
  });

  it("the design really has duplicate rows (n_unique < n_expanded)", () => {
    expect(design.nExpanded).toBe(40 * (3 + 1));
    expect(design.samples.length / design.nParams).toBeLessThan(design.nExpanded);
  });

  it("gather path (unique y + expandedToUnique) equals the direct expanded computation", () => {
    const gathered = analyzeMorris(design, yUnique);
    console.log("gathered mu       =", Array.from(gathered.mu));
    console.log("gathered mu_star  =", Array.from(gathered.mu_star));
    console.log("gathered sigma    =", Array.from(gathered.sigma));
    console.log("direct   mu_star  =", Array.from(directMuStar));
    console.log("direct   sigma    =", Array.from(directSigma));
    expect(allclose(gathered.mu, Array.from(directMu), { atol: 1e-12, rtol: 0 })).toBe(true);
    expect(allclose(gathered.mu_star, Array.from(directMuStar), { atol: 1e-12, rtol: 0 })).toBe(
      true,
    );
    expect(allclose(gathered.sigma, Array.from(directSigma), { atol: 1e-12, rtol: 0 })).toBe(
      true,
    );
  });
});

// ---------------------------------------------------------------------------
// Sampler structure
// ---------------------------------------------------------------------------

describe("sampleMorris trajectory sampler structure", () => {
  const design = sampleMorris(ISHIGAMI_PROBLEM, 40, { numLevels: 4, seed: 1, verbose: false });
  const D = design.nParams;

  it("nExpanded == r * (D + 1) and ee bookkeeping is (r, D) flat", () => {
    expect(design.nExpanded).toBe(design.nTrajectories * (D + 1));
    expect(design.eeIdxAfter.length).toBe(design.nTrajectories * D);
    expect(design.eeIdxBefore.length).toBe(design.nTrajectories * D);
    expect(design.eeDelta.length).toBe(design.nTrajectories * D);
  });

  it("ee index arrays stay within [0, nExpanded) and delta is ±p / (2 (p - 1))", () => {
    const p = design.numLevels;
    for (let i = 0; i < design.eeIdxAfter.length; i++) {
      expect(design.eeIdxAfter[i]).toBeGreaterThanOrEqual(0);
      expect(design.eeIdxAfter[i]).toBeLessThan(design.nExpanded);
      expect(design.eeIdxBefore[i]).toBeGreaterThanOrEqual(0);
      expect(design.eeIdxBefore[i]).toBeLessThan(design.nExpanded);
      expect(design.eeIdxBefore[i]).toBeLessThan(design.eeIdxAfter[i]);
    }
    for (let i = 0; i < design.eeDelta.length; i++) {
      expect(Math.abs(Math.abs(design.eeDelta[i]) - p / (2 * (p - 1)))).toBeLessThan(1e-15);
    }
  });

  it("expandedToUnique maps each expanded row to [0, nUnique) and dedup removed rows", () => {
    const nUnique = design.samples.length / D;
    expect(design.expandedToUnique.length).toBe(design.nExpanded);
    for (let i = 0; i < design.expandedToUnique.length; i++) {
      expect(design.expandedToUnique[i]).toBeGreaterThanOrEqual(0);
      expect(design.expandedToUnique[i]).toBeLessThan(nUnique);
    }
    // 160 rows over the 4^3 grid: duplicates are guaranteed.
    expect(nUnique).toBeLessThan(design.nExpanded);
  });

  it("same seed -> byte-identical designs; different seeds -> differ", () => {
    const a = sampleMorris(ISHIGAMI_PROBLEM, 40, { numLevels: 4, seed: 1, verbose: false });
    const b = sampleMorris(ISHIGAMI_PROBLEM, 40, { numLevels: 4, seed: 1, verbose: false });
    const c = sampleMorris(ISHIGAMI_PROBLEM, 40, { numLevels: 4, seed: 2, verbose: false });
    expect(a.samples).toEqual(b.samples);
    expect(a.expandedToUnique).toEqual(b.expandedToUnique);
    expect(a.eeIdxAfter).toEqual(b.eeIdxAfter);
    expect(a.eeIdxBefore).toEqual(b.eeIdxBefore);
    expect(a.eeDelta).toEqual(b.eeDelta);
    expect(a.samples).not.toEqual(c.samples);
  });

  it("all samples stay within the declared uniform bounds [-pi, pi]", () => {
    for (let i = 0; i < design.samples.length; i++) {
      expect(design.samples[i]).toBeGreaterThanOrEqual(-Math.PI);
      expect(design.samples[i]).toBeLessThanOrEqual(Math.PI);
    }
  });

  it("raises a clear error for method='radial' (not ported)", () => {
    expect(() =>
      sampleMorris(ISHIGAMI_PROBLEM, 10, { method: "radial" }),
    ).toThrow(/radial/);
  });

  it("raises for invalid arguments (trajectories < 2, num_levels < 2, categoricals)", () => {
    expect(() => sampleMorris(ISHIGAMI_PROBLEM, 1)).toThrow(/n_trajectories must be >= 2/);
    expect(() => sampleMorris(ISHIGAMI_PROBLEM, 10, { numLevels: 1 })).toThrow(
      /num_levels must be >= 2/,
    );
    const catProblem: ProblemSpec = {
      names: ["c"],
      marginals: [{ kind: "categorical", probs: [0.5, 0.5] }],
    };
    expect(() => sampleMorris(catProblem, 10)).toThrow(/categorical/);
  });
});

// ---------------------------------------------------------------------------
// Statistical sanity (JS sampler end-to-end)
// ---------------------------------------------------------------------------

describe("analyzeMorris statistical sanity on Ishigami (r=100)", () => {
  let mu_star: Float64Array;
  let mu: Float64Array;
  let sigma: Float64Array;

  beforeAll(async () => {
    await init();
    defaultDevice("wasm");
    const design = sampleMorris(ISHIGAMI_PROBLEM, 100, {
      numLevels: 4,
      seed: 3,
      verbose: false,
    });
    const yUnique = evaluate(design);
    ({ mu, mu_star, sigma } = analyzeMorris(design, yUnique));
    console.log("sanity mu       =", Array.from(mu));
    console.log("sanity mu_star  =", Array.from(mu_star));
    console.log("sanity sigma    =", Array.from(sigma));
  });

  it("mu_star ranks x2 highest and x1 above x3 (x2 = max, x1 > x3)", () => {
    expect(mu_star[1]).toBeGreaterThan(mu_star[0]);
    expect(mu_star[1]).toBeGreaterThan(mu_star[2]);
    expect(mu_star[0]).toBeGreaterThan(mu_star[2]);
  });

  it("all measures are finite", () => {
    for (const v of mu) expect(Number.isFinite(v)).toBe(true);
    for (const v of mu_star) expect(Number.isFinite(v)).toBe(true);
    for (const v of sigma) expect(Number.isFinite(v)).toBe(true);
  });
});