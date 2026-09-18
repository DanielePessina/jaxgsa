// Direction numbers for the Sobol' sequence, dims 1..40, 32 bits.
//
// Source: the canonical published table "new-joe-kuo-6.21201" (Joe & Kuo,
// https://web.maths.unsw.edu.au/~fkuo/sobol/), rows labelled 2..40, each
// transcribed verbatim as `[d, s, a, m1 .. ms]` where:
//   d  Sobol' dimension this row serves (label d in the published file)
//   s  degree of the primitive polynomial
//   a  polynomial coefficient integer (bits c_1..c_{s-1}; the full
//      polynomial is x^s + c_1 x^(s-1) + ... + c_{s-1} x + 1)
//   m  the s initial direction-number integers m_1 .. m_s (odd, 0 < m_i < 2^i)
//
// Dimension 1 has no published row: it is the base-2 van der Corput sequence,
// whose direction numbers are all 1 (v[i] = 1 << (32 - i)). Its row is
// hard-coded in buildV() below. Dimensions 2..40 use the rows below; the
// published file's row labelled d serves Sobol' dimension d (labels 1 is
// absent and dim 1 is degenerate-identical to the label-2 row in (s, m)
// terms but NOT in the direction numbers, so it must not be derived from it).
//
// The full 32-bit direction numbers per dimension are computed once at
// module load with the Bratley & Fox (1988) recurrence, exactly as scipy
// does in scipy/stats/_sobol.pyx `_initialize_v`: the transcription above
// was validated to reproduce scipy's direction-number matrix bit-for-bit for
// dims 1..40.
export const SOBOL_MAX_DIM = 40;

export const SOBOL_BITS = 32;

/** Raw published rows for dims 2..40: [d, s, a, m1, ..., ms]. */
export const JOE_KUO_ROWS: ReadonlyArray<readonly number[]> = [
  // dim 2: s=1, a=0, m=1
  [2, 1, 0, 1],
  // dim 3: s=2, a=1, m=1 3
  [3, 2, 1, 1, 3],
  // dim 4: s=3, a=1, m=1 3 1
  [4, 3, 1, 1, 3, 1],
  // dim 5: s=3, a=2, m=1 1 1
  [5, 3, 2, 1, 1, 1],
  // dim 6: s=4, a=1, m=1 1 3 3
  [6, 4, 1, 1, 1, 3, 3],
  // dim 7: s=4, a=4, m=1 3 5 13
  [7, 4, 4, 1, 3, 5, 13],
  // dim 8: s=5, a=2, m=1 1 5 5 17
  [8, 5, 2, 1, 1, 5, 5, 17],
  // dim 9: s=5, a=4, m=1 1 5 5 5
  [9, 5, 4, 1, 1, 5, 5, 5],
  // dim 10: s=5, a=7, m=1 1 7 11 19
  [10, 5, 7, 1, 1, 7, 11, 19],
  // dim 11: s=5, a=11, m=1 1 5 1 1
  [11, 5, 11, 1, 1, 5, 1, 1],
  // dim 12: s=5, a=13, m=1 1 1 3 11
  [12, 5, 13, 1, 1, 1, 3, 11],
  // dim 13: s=5, a=14, m=1 3 5 5 31
  [13, 5, 14, 1, 3, 5, 5, 31],
  // dim 14: s=6, a=1, m=1 3 3 9 7 49
  [14, 6, 1, 1, 3, 3, 9, 7, 49],
  // dim 15: s=6, a=13, m=1 1 1 15 21 21
  [15, 6, 13, 1, 1, 1, 15, 21, 21],
  // dim 16: s=6, a=16, m=1 3 1 13 27 49
  [16, 6, 16, 1, 3, 1, 13, 27, 49],
  // dim 17: s=6, a=19, m=1 1 1 15 7 5
  [17, 6, 19, 1, 1, 1, 15, 7, 5],
  // dim 18: s=6, a=22, m=1 3 1 15 13 25
  [18, 6, 22, 1, 3, 1, 15, 13, 25],
  // dim 19: s=6, a=25, m=1 1 5 5 19 61
  [19, 6, 25, 1, 1, 5, 5, 19, 61],
  // dim 20: s=7, a=1, m=1 3 7 11 23 15 103
  [20, 7, 1, 1, 3, 7, 11, 23, 15, 103],
  // dim 21: s=7, a=4, m=1 3 7 13 13 15 69
  [21, 7, 4, 1, 3, 7, 13, 13, 15, 69],
  // dim 22: s=7, a=7, m=1 1 3 13 7 35 63
  [22, 7, 7, 1, 1, 3, 13, 7, 35, 63],
  // dim 23: s=7, a=8, m=1 3 5 9 1 25 53
  [23, 7, 8, 1, 3, 5, 9, 1, 25, 53],
  // dim 24: s=7, a=14, m=1 3 1 13 9 35 107
  [24, 7, 14, 1, 3, 1, 13, 9, 35, 107],
  // dim 25: s=7, a=19, m=1 3 1 5 27 61 31
  [25, 7, 19, 1, 3, 1, 5, 27, 61, 31],
  // dim 26: s=7, a=21, m=1 1 5 11 19 41 61
  [26, 7, 21, 1, 1, 5, 11, 19, 41, 61],
  // dim 27: s=7, a=28, m=1 3 5 3 3 13 69
  [27, 7, 28, 1, 3, 5, 3, 3, 13, 69],
  // dim 28: s=7, a=31, m=1 1 7 13 1 19 1
  [28, 7, 31, 1, 1, 7, 13, 1, 19, 1],
  // dim 29: s=7, a=32, m=1 3 7 5 13 19 59
  [29, 7, 32, 1, 3, 7, 5, 13, 19, 59],
  // dim 30: s=7, a=37, m=1 1 3 9 25 29 41
  [30, 7, 37, 1, 1, 3, 9, 25, 29, 41],
  // dim 31: s=7, a=41, m=1 3 5 13 23 1 55
  [31, 7, 41, 1, 3, 5, 13, 23, 1, 55],
  // dim 32: s=7, a=42, m=1 3 7 3 13 59 17
  [32, 7, 42, 1, 3, 7, 3, 13, 59, 17],
  // dim 33: s=7, a=50, m=1 3 1 3 5 53 69
  [33, 7, 50, 1, 3, 1, 3, 5, 53, 69],
  // dim 34: s=7, a=55, m=1 1 5 5 23 33 13
  [34, 7, 55, 1, 1, 5, 5, 23, 33, 13],
  // dim 35: s=7, a=56, m=1 1 7 7 1 61 123
  [35, 7, 56, 1, 1, 7, 7, 1, 61, 123],
  // dim 36: s=7, a=59, m=1 1 7 9 13 61 49
  [36, 7, 59, 1, 1, 7, 9, 13, 61, 49],
  // dim 37: s=7, a=62, m=1 3 3 5 3 55 33
  [37, 7, 62, 1, 3, 3, 5, 3, 55, 33],
  // dim 38: s=8, a=14, m=1 3 1 15 31 13 49 245
  [38, 8, 14, 1, 3, 1, 15, 31, 13, 49, 245],
  // dim 39: s=8, a=21, m=1 3 5 15 31 59 63 97
  [39, 8, 21, 1, 3, 5, 15, 31, 59, 63, 97],
  // dim 40: s=8, a=22, m=1 3 1 11 11 11 77 249
  [40, 8, 22, 1, 3, 1, 11, 11, 11, 77, 249],
];

/** Bratley-Fox / scipy-identical recurrence. Returns v[d] for Sobol' dim `dim`. */
export function buildDirectionNumbers(dim: number): Uint32Array {
  const v = new Uint32Array(SOBOL_BITS);

  if (dim < 1 || dim > SOBOL_MAX_DIM) {
    throw new RangeError(`Sobol' dimension ${dim} out of range [1, ${SOBOL_MAX_DIM}]`);
  }

  if (dim === 1) {
    // van der Corput: every direction number is 1 << (32 - i).
    let pow = 1 << 31;

    for (let j = 0; j < SOBOL_BITS; j++) {
      v[j] = pow >>> 0;
      pow >>>= 1;
    }

    return v;
  }

  const [, s, a, ...ms] = JOE_KUO_ROWS[dim - 2];
  const p = 2 * a + (1 << s) + 1;

  for (let i = 0; i < s; i++) v[i] = ms[i];

  for (let j = s; j < SOBOL_BITS; j++) {
    let newv = v[j - s];
    let pow2 = 1;

    for (let k = 0; k < s; k++) {
      pow2 <<= 1;

      if (((p >>> (s - 1 - k)) & 1) === 1) {
        newv = (newv ^ Math.imul(pow2, v[j - k - 1])) >>> 0;
      }
    }

    v[j] = newv;
  }

  // Scale column j by 2^(31-j) mod 2^32 (scipy's final multiply step).
  for (let j = 0; j < SOBOL_BITS; j++) {
    v[j] = Math.imul(v[j], 1 << (SOBOL_BITS - 1 - j)) >>> 0;
  }

  return v;
}

let cached: Uint32Array[] | null = null;

/** Cached direction-number rows for dims 1..SOBOL_MAX_DIM. */
export function directionNumbers(): Uint32Array[] {
  if (cached === null) {
    cached = new Array(SOBOL_MAX_DIM);

    for (let d = 1; d <= SOBOL_MAX_DIM; d++) cached[d - 1] = buildDirectionNumbers(d);
  }

  return cached;
}
