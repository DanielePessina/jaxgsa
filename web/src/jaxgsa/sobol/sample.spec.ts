import { describe, expect, it } from "vitest";
import type { ProblemSpec } from "./sample";
import { buildExpandedSamples, sample, saltelliStep, transformValue } from "./sample";
import { sobolSequence } from "./sampler";

// ---------------------------------------------------------------------------
// Test problem: 3 independent uniforms on [-pi, pi] (the classic Saltelli
// demo problem).
// ---------------------------------------------------------------------------

const PROBLEM: ProblemSpec = {
  names: ["x1", "x2", "x3"],
  marginals: [
    { kind: "uniform", low: -Math.PI, high: Math.PI },
    { kind: "uniform", low: -Math.PI, high: Math.PI },
    { kind: "uniform", low: -Math.PI, high: Math.PI },
  ],
};

const PI2 = 2 * Math.PI;

function sample2D(seq: Float64Array, dim: number, index: number, col: number): number {
  return seq[(index - 1) * dim + col];
}

describe("Sobol' sequence machinery (van der Corput gate)", () => {
  it("dimension 1, unscrambled, is exactly the base-2 van der Corput sequence", () => {
    const seq = sobolSequence(1, 8, false, 0);
    const expected = [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625];

    for (let i = 0; i < 8; i++) expect(seq[i]).toBe(expected[i]);
  });

  it("matches scipy-derived goldens (dims 1..40, unscrambled)", () => {
    // Values computed from scipy.stats._sobol direction numbers with the
    // same direct (bit-reversal) construction the port uses; dyadic
    // rationals, hence exact float64 comparisons.
    const dims = [1, 2, 3, 5, 8, 13, 21, 34, 40];
    const indices = [2, 3, 8, 16, 127, 256, 512, 2048, 4096];

    const golden = [
      [0.25, 0.75, 0.0625, 0.03125, 0.9921875, 0.001953125, 0.0009765625, 0.000244140625, 0.0001220703125],
      [0.75, 0.25, 0.9375, 0.53125, 0.9921875, 0.501953125, 0.7529296875, 0.941162109375, 0.5333251953125],
      [0.75, 0.25, 0.5625, 0.90625, 0.5390625, 0.408203125, 0.6123046875, 0.334228515625, 0.5015869140625],
      [0.25, 0.75, 0.6875, 0.96875, 0.0546875, 0.353515625, 0.1865234375, 0.940185546875, 0.8536376953125],
      [0.25, 0.75, 0.3125, 0.53125, 0.5546875, 0.462890625, 0.6181640625, 0.390869140625, 0.6959228515625],
      [0.75, 0.25, 0.3125, 0.96875, 0.1484375, 0.060546875, 0.1298828125, 0.368896484375, 0.8438720703125],
      [0.75, 0.25, 0.8125, 0.40625, 0.6796875, 0.662109375, 0.8662109375, 0.225830078125, 0.5858154296875],
      [0.25, 0.75, 0.3125, 0.71875, 0.0859375, 0.740234375, 0.8740234375, 0.262939453125, 0.8524169921875],
      [0.75, 0.25, 0.6875, 0.34375, 0.2421875, 0.236328125, 0.6982421875, 0.227294921875, 0.5374755859375],
    ] as const;

    for (let d = 0; d < dims.length; d++) {
      const dim = dims[d];
      const seq = sobolSequence(dim, 4096, false, 0);

      for (let k = 0; k < indices.length; k++) {
        const i = indices[k];
        expect(sample2D(seq, dim, i, dim - 1)).toBe(golden[d][k]);
      }
    }
  });

  it("rejects dims beyond the 40-entry table and counts beyond the cap", () => {
    expect(() => sobolSequence(41, 8, false, 0)).toThrow(/out of range/);
    expect(() => sobolSequence(0, 8, false, 0)).toThrow(/out of range/);
    expect(() => sobolSequence(2, 2 ** 24 + 1, false, 0)).toThrow(/exceeds the cap/);
  });
});

describe("scrambling (affine linear scramble + digital shift)", () => {
  it("is deterministic: same seed -> byte-identical sequences", () => {
    const a = sobolSequence(5, 512, true, 12345);
    const b = sobolSequence(5, 512, true, 12345);
    expect(a.length).toBe(b.length);

    for (let i = 0; i < a.length; i++) expect(a[i]).toBe(b[i]);
  });

  it("differs across seeds", () => {
    const a = sobolSequence(5, 64, true, 1);
    const b = sobolSequence(5, 64, true, 2);
    let anyDifferent = false;

    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) anyDifferent = true;
    }

    expect(anyDifferent).toBe(true);
  });

  it("keeps values in [0, 1) and ignores the seed when unscrambled", () => {
    for (const seed of [0, 7, 424242]) {
      const seq = sobolSequence(3, 256, true, seed);

      for (let i = 0; i < seq.length; i++) {
        expect(seq[i]).toBeGreaterThanOrEqual(0);
        expect(seq[i]).toBeLessThan(1);
      }

      const unscrambledA = sobolSequence(3, 64, false, 0);
      const unscrambledB = sobolSequence(3, 64, false, 99);

      for (let i = 0; i < unscrambledA.length; i++) {
        expect(unscrambledA[i]).toBe(unscrambledB[i]);
      }
    }
  });
});

describe("Saltelli layout", () => {
  const D = 3;
  const baseN = 512;
  const seed = 7;

  it("expanded size follows n_params * (2*D + 2) or n_params + 2 per base point", () => {
    expect(saltelliStep(3, false)).toBe(5);
    expect(saltelliStep(3, true)).toBe(8);
    expect(saltelliStep(10, true)).toBe(22);
  });

  it("builds the A/AB/BA/B interleaved layout row-major", () => {
    const raw = buildExpandedSamples(D, baseN, {
      calcSecondOrder: false,
      scramble: true,
      seed,
    });

    // The underlying draw is a 2D-dimensional sequence; A uses the first D
    // columns and B the last D.
    const base = sobolSequence(2 * D, baseN, true, seed);
    expect(raw.length).toBe(baseN * 5 * D);

    for (const i of [0, 1, 7, 100, 511]) {
      for (let j = 0; j < D; j++) {
        expect(raw[i * 5 * D + j]).toBe(base[i * 2 * D + j]); // A_i
        expect(raw[i * 5 * D + 4 * D + j]).toBe(base[i * 2 * D + D + j]); // B_i

        for (let jj = 0; jj < D; jj++) {
          // AB_jj = A with column jj replaced by B's column jj
          const ab = raw[i * 5 * D + (1 + jj) * D + j];
          const expected = j === jj ? base[i * 2 * D + D + jj] : base[i * 2 * D + j];
          expect(ab).toBe(expected);
        }
      }
    }
  });

  it("public design exposes the expanded count and consistent dedup maps", () => {
    const design = sample(PROBLEM, 1, {
      baseN,
      calcSecondOrder: false,
      scramble: true,
      seed,
      verbose: false,
    });

    expect(design.baseN).toBe(baseN);
    expect(design.nParams).toBe(D);
    expect(design.calcSecondOrder).toBe(false);
    expect(design.nExpanded).toBe(baseN * 5);
    const nRuns = design.samples.length / D;
    expect(nRuns).toBeLessThanOrEqual(design.nExpanded);
    expect(design.expandedToUnique.length).toBe(design.nExpanded);

    // Every expanded row maps to a unique row that is bitwise equal to its
    // marginal transform, and rows keep first-occurrence order.
    const raw = buildExpandedSamples(D, baseN, {
      calcSecondOrder: false,
      scramble: true,
      seed,
    });

    const seen = new Set<number>();
    let expectFirstOccurrence = true;

    for (let r = 0; r < design.nExpanded; r++) {
      const u = design.expandedToUnique[r];
      expect(u).toBeGreaterThanOrEqual(0);
      expect(u).toBeLessThan(nRuns);

      for (let j = 0; j < D; j++) {
        const physical = transformValue(raw[r * D + j], PROBLEM.marginals[j]);
        expect(design.samples[u * D + j]).toBe(physical);
      }

      if (expectFirstOccurrence && seen.has(u)) expectFirstOccurrence = false;
      seen.add(u);
    }

    expect(seen.size).toBe(nRuns);

    // A_1 == B_1 for the unscrambled design (every dim starts at x_1 = 0.5):
    // the first base point's A and B rows must dedup onto the same row.
    const plain = sample(PROBLEM, 1, {
      baseN: 16,
      calcSecondOrder: false,
      scramble: false,
      seed: 0,
      verbose: false,
    });

    const rawPlain = buildExpandedSamples(D, 16, {
      calcSecondOrder: false,
      scramble: false,
      seed: 0,
    });

    for (let j = 0; j < D; j++) {
      expect(rawPlain[0 * D + j]).toBe(rawPlain[4 * D + j]); // A_1 == B_1
      expect(plain.samples[plain.expandedToUnique[0] * D + j]).toBe(
        plain.samples[plain.expandedToUnique[4] * D + j],
      );
    }

    expect(plain.samples.length / D).toBeLessThan(plain.nExpanded);
  });
});

describe("marginal transforms", () => {
  it("keeps unit-cube values in [0, 1) and transformed uniforms inside the bounds", () => {
    const design = sample(PROBLEM, 1, {
      baseN: 128,
      calcSecondOrder: false,
      scramble: true,
      seed: 3,
      verbose: false,
    });

    const raw = buildExpandedSamples(3, 128, {
      calcSecondOrder: false,
      scramble: true,
      seed: 3,
    });

    for (let i = 0; i < raw.length; i++) {
      expect(raw[i]).toBeGreaterThanOrEqual(0);
      expect(raw[i]).toBeLessThan(1);
    }

    for (let i = 0; i < design.samples.length; i++) {
      expect(design.samples[i]).toBeGreaterThanOrEqual(-Math.PI);
      expect(design.samples[i]).toBeLessThan(Math.PI);
    }
  });

  it("gives empirically sane columns: mean ~ 0, variance ~ (2*pi)^2/12", () => {
    const design = sample(PROBLEM, 1, {
      baseN: 512,
      calcSecondOrder: false,
      scramble: true,
      seed: 11,
      verbose: false,
    });

    const D = 3;
    const nRuns = design.samples.length / D;

    for (let j = 0; j < D; j++) {
      let mean = 0;

      for (let r = 0; r < nRuns; r++) mean += design.samples[r * D + j];
      mean /= nRuns;
      expect(Math.abs(mean)).toBeLessThan(0.15);
      let variance = 0;

      for (let r = 0; r < nRuns; r++) {
        const d = design.samples[r * D + j] - mean;
        variance += d * d;
      }

      variance /= nRuns - 1;
      expect(Math.abs(variance - (PI2 * PI2) / 12)).toBeLessThan(0.5);
    }
  });

  it("gaussian marginals stay near their moments and respect truncation", () => {
    const gaussProblem: ProblemSpec = {
      names: ["g"],
      marginals: [{ kind: "gaussian", mean: 2, variance: 4, low: -10, high: 10 }],
    };

    const raw = buildExpandedSamples(1, 4096, {
      calcSecondOrder: false,
      scramble: true,
      seed: 5,
    });

    let mean = 0;

    for (let i = 0; i < raw.length; i++) mean += transformValue(raw[i], gaussProblem.marginals[0]);
    mean /= raw.length;
    expect(Math.abs(mean - 2)).toBeLessThan(0.2);
    let anyTruncated = false;

    for (let i = 0; i < raw.length; i++) {
      const v = transformValue(raw[i], gaussProblem.marginals[0]);
      expect(v).toBeGreaterThanOrEqual(-10);
      expect(v).toBeLessThanOrEqual(10);

      if (v === -10 || v === 10) anyTruncated = true;
    }

    expect(anyTruncated).toBe(false); // truncation window is generous: no clamping mass
  });

  it("raises a clear error for categorical marginals (not yet ported)", () => {
    const catProblem: ProblemSpec = {
      names: ["c"],
      marginals: [{ kind: "categorical", probs: [0.5, 0.5] }],
    };

    expect(() => transformValue(0.3, catProblem.marginals[0])).toThrow(/not yet ported/);
    expect(() => sample(catProblem, 1, { baseN: 8, verbose: false })).toThrow(/not yet ported/);
  });
});

describe("sample(): seed handling, validation, doubling loop", () => {
  it("same seed -> byte-identical designs; different seeds -> different A matrices", () => {
    const opts = { baseN: 256, calcSecondOrder: true, scramble: true, verbose: false } as const;
    const a = sample(PROBLEM, 1, { ...opts, seed: 42 });
    const b = sample(PROBLEM, 1, { ...opts, seed: 42 });
    const c = sample(PROBLEM, 1, { ...opts, seed: 43 });
    expect(a.samples.length).toBe(b.samples.length);

    for (let i = 0; i < a.samples.length; i++) expect(a.samples[i]).toBe(b.samples[i]);
    expect(Array.from(a.expandedToUnique)).toEqual(Array.from(b.expandedToUnique));

    // The A matrix of each design is the first baseN block rows; it differs
    // across seeds.
    const rawA = buildExpandedSamples(3, 256, { ...opts, seed: 42 });
    const rawC = buildExpandedSamples(3, 256, { ...opts, seed: 43 });
    const aLen = 256 * 3;
    let anyDifferent = false;

    for (let i = 0; i < aLen; i++) {
      if (rawA[i] !== rawC[i]) anyDifferent = true;
    }

    expect(anyDifferent).toBe(true);
    expect(Array.from(a.samples)).not.toEqual(Array.from(c.samples));
  });

  it("rejects a non-power-of-2 base_n", () => {
    expect(() =>
      sample(PROBLEM, 1, { baseN: 100, verbose: false }),
    ).toThrow(/power of 2/);
    expect(() =>
      sample(PROBLEM, 1, { baseN: 0, verbose: false }),
    ).toThrow(/power of 2/);
    expect(() => sample(PROBLEM, 1, { baseN: 512, verbose: false })).not.toThrow();
  });

  it("rejects correlated problems", () => {
    expect(() =>
      sample(
        { names: PROBLEM.names, marginals: PROBLEM.marginals, correlation: { kind: "pearson" } },
        1,
        { baseN: 32, verbose: false },
      ),
    ).toThrow(/correlat/i);
  });

  it("rejects more than 20 parameters", () => {
    const names = Array.from({ length: 21 }, (_, i) => `x${i}`);

    const marginals: ProblemSpec["marginals"] = names.map(() => ({
      kind: "uniform",
      low: 0,
      high: 1,
    }));

    expect(() => sample({ names, marginals }, 1, { baseN: 8, verbose: false })).toThrow(
      /at most 20/,
    );
  });

  it("n_samples path returns at least the requested unique rows (doubling loop)", () => {
    const design = sample(PROBLEM, 500, { scramble: true, seed: 1, verbose: false });
    const D = design.nParams;
    const nRuns = design.samples.length / D;
    expect(nRuns).toBeGreaterThanOrEqual(500);
    expect(design.nExpanded).toBe(design.baseN * saltelliStep(D, design.calcSecondOrder));
    expect(Number.isInteger(Math.log2(design.baseN))).toBe(true);
  });
});