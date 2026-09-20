/**
 * Analytic test functions shared by the method ports, matching the Python
 * `jaxgsa.benchmarks` module.
 */

/**
 * Ishigami function over [-pi, pi]^3 (x1, x2, x3 = indices 0, 1, 2):
 *   f(x) = sin(x1) + 7 sin(x2)^2 + 0.1 x3^4 sin(x1)
 */
export function ishigami(x: number[]): number {
  return (
    Math.sin(x[0]) + 7 * Math.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * Math.sin(x[0])
  );
}

/**
 * Linear additive model over [0, 1]^3: f(x) = x1 + 2 x2 + 3 x3. Purely
 * additive, so S1 == ST and all interactions vanish.
 */
export function linear(x: number[]): number {
  return x[0] + 2 * x[1] + 3 * x[2];
}

/**
 * Gaussian linear model over standard normals: f(x) = x1 + 2 x2 + 3 x3.
 * Same index structure as `linear` with gaussian marginals.
 */
export function gaussianLinear(x: number[]): number {
  return linear(x);
}

/**
 * Sobol' G-function over [0, 1]^8:
 *   f(x) = prod_j (|4 x_j - 2| + a_j) / (1 + a_j)
 * with a = (0, 1, 4.5, 9, 99, 99, 99, 99). Multiplicative — every subset of
 * inputs interacts; a_j = 0 makes x_j maximally influential.
 */
export function sobolG(x: number[]): number {
  const a = [0, 1, 4.5, 9, 99, 99, 99, 99];
  let p = 1;

  for (let j = 0; j < x.length; j++) {
    p *= (Math.abs(4 * x[j] - 2) + a[j]) / (1 + a[j]);
  }

  return p;
}