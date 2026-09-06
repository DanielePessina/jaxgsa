import { directionNumbers, SOBOL_BITS, SOBOL_MAX_DIM } from "./directions";

/**
 * Maximum number of base points: the direction numbers carry 32 bits, and the
 * incremental generation below is exact for any count up to 2^32, but the
 * design layer caps base_n at 2^24 (a quarter of the addressable range, far
 * beyond any realistic sampling budget) to keep allocations and the doubling
 * loop in check.
 */
export const MAX_SOBOL_POINTS = 1 << 24;

const TWO_32 = 4294967296;

/**
 * splitmix32: a tiny deterministic 32-bit PRNG (the standard 32-bit variant
 * of splitmix64, public-domain). Used only to derive per-dimension scramble
 * parameters from a seed; same seed -> same stream.
 */
export function splitmix32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x9e3779b9) | 0;
    let t = a ^ (a >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t = t ^ (t >>> 15);
    t = Math.imul(t, 0x735a2d97);
    return (t = t ^ (t >>> 15)) >>> 0;
  };
}

/** Derive the (mask, shift) pair for dimension `dim` from `seed`. */
export function scrambleParams(seed: number, dim: number): [number, number] {
  // Mix the seed with the dimension so different dims get independent
  // parameters even for seed 0, then warm up the stream.
  const g = splitmix32((seed >>> 0) ^ Math.imul(dim + 1, 0x9e3779b9));
  g();
  const mask = g();
  const shift = g();
  return [mask, shift];
}

/**
 * Scrambled Sobol' sequence, dims 1..40, row-major Float64Array of
 * `count * dim` values in [0, 1).
 *
 * Points are generated with the standard "bit-reversal" (direct) form: the
 * i-th point (1-indexed, i = 1, 2, ...) in dimension d is the XOR of the
 * direction numbers v[d][k] over the set bits k of i, divided by 2^32. In
 * dimension 1 (unscrambled) this is exactly the base-2 van der Corput
 * sequence 1/2, 1/4, 3/4, 1/8, 5/8, ... The equivalent O(1)-per-point
 * incremental "Gray code" form x[i] = x[i-1] ^ v[d][c] (c = index of the
 * lowest zero bit of i-1) generates the *same point set* in a permuted
 * order; the direct form is used here because the van der Corput ordering is
 * the canonical definition and the correctness gate for the whole sequence
 * machinery. Cost is ~popcount(i) XORs per point per dim, which is cheap at
 * the base_n sizes this sampler is used for (<= 2^20 in the doubling loop).
 *
 * Scrambling (when `scramble` is true) is a seeded affine randomization per
 * dimension: `mask` (a bitwise linear scramble, i.e. XOR with a constant
 * 32-bit vector) and `shift` (a digital shift, an additive constant modulo
 * 2^32), both drawn from splitmix32 seeded with (seed, dim). The scrambled
 * value is ((x XOR mask) + shift) mod 1, computed in 32-bit then mapped to
 * [0, 1). This is a valid QMC randomization (a linear scramble + digital
 * shift, in the family described by Owen 1997): it preserves the
 * equidistribution of the sequence while destroying its seed-independent
 * structure, so different seeds give statistically independent designs.
 * It is deliberately NOT the full Owen/Nested-uniform-scramble that
 * scipy.stats.qmc applies — output values for scrambled draws therefore
 * differ from scipy, and only the unscrambled (scramble=false) sequence is
 * bit-comparable to scipy.
 */
export function sobolSequence(
  dim: number,
  count: number,
  scramble: boolean,
  seed: number,
): Float64Array {
  if (!Number.isInteger(dim) || dim < 1 || dim > SOBOL_MAX_DIM) {
    throw new RangeError(
      `Sobol' sequence dimension ${dim} out of range [1, ${SOBOL_MAX_DIM}]`,
    );
  }
  if (!Number.isInteger(count) || count < 1) {
    throw new RangeError(`Sobol' point count must be a positive integer (got ${count})`);
  }
  if (count > MAX_SOBOL_POINTS) {
    throw new RangeError(
      `Sobol' point count ${count} exceeds the cap of 2^24 = ${MAX_SOBOL_POINTS} ` +
        `(direction numbers carry ${SOBOL_BITS} bits)`,
    );
  }

  const v = directionNumbers();
  const out = new Float64Array(count * dim);
  const masks = new Uint32Array(dim);
  const shifts = new Uint32Array(dim);
  if (scramble) {
    for (let d = 0; d < dim; d++) {
      const [mask, shift] = scrambleParams(seed, d + 1);
      masks[d] = mask;
      shifts[d] = shift;
    }
  }

  const acc = new Uint32Array(dim);
  for (let i = 1; i <= count; i++) {
    const base = (i - 1) * dim;
    let n = i;
    // Accumulate, per dimension, the XOR of v[d][k] over the set bits k of i.
    acc.fill(0);
    while (n !== 0) {
      const lsb = n & -n;
      const k = 31 - Math.clz32(lsb);
      for (let d = 0; d < dim; d++) acc[d] ^= v[d][k];
      n &= n - 1;
    }
    for (let d = 0; d < dim; d++) {
      let x = acc[d];
      if (scramble) x = ((x ^ masks[d]) + shifts[d]) >>> 0;
      out[base + d] = x / TWO_32;
    }
  }
  return out;
}