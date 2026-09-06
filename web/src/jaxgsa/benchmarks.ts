/**
 * Analytic test functions shared by the method ports, matching the Python
 * `benchmark_salib.py` / jaxgsa example functions.
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