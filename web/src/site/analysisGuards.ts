/**
 * Pure checks the Analyze step runs before and after a fit: the PCE order
 * field, the run token that drops stale results, and the fit-quality and
 * low-N messages shown on result cards. They have no React or wasm
 * dependencies, so they are cheap to test.
 */

import type { PceDiagnostics } from "@/jaxgsa/pce/analyze";

/** `POORFIT_THRESHOLD` in `src/jaxgsa/_core/fit_quality.py`. */
export const POORFIT_THRESHOLD = 0.5;

/**
 * `LOO_RATIO_THRESHOLD` in `fit_quality.py`: the leave-one-out R-squared
 * falls to `POORFIT_THRESHOLD` when `cv_rmse / std(Y)` reaches this value.
 */
export const LOO_RATIO_THRESHOLD = Math.sqrt(1 - POORFIT_THRESHOLD);

export type OrderParse =
  | { ok: true; value: number }
  | { ok: false; message: string };

/**
 * Read the PCE order field. The order must be a whole number of at least 1
 * (Python raises `order must be >= 1`). An empty field is an error, not 0.
 */
export function parseOrder(text: string): OrderParse {
  const trimmed = text.trim();

  if (trimmed === "") {
    return { ok: false, message: "Enter a PCE order. It must be a whole number of 1 or more." };
  }

  if (!/^[+-]?\d+$/.test(trimmed)) {
    return { ok: false, message: "The PCE order must be a whole number of 1 or more." };
  }

  const value = Number(trimmed);

  if (!Number.isSafeInteger(value) || value < 1) {
    return { ok: false, message: "The PCE order must be a whole number of 1 or more." };
  }

  return { ok: true, value };
}

/**
 * A run token. `start()` begins a run and returns its token. `cancel()`
 * makes every earlier token stale. A loop that awaits between methods checks
 * `isCurrent(token)` after each await and drops its result when it is stale.
 */
export interface RunGuard {
  start: () => number;
  cancel: () => void;
  isCurrent: (token: number) => boolean;
}

export function createRunGuard(): RunGuard {
  let current = 0;

  return {
    start: () => ++current,
    cancel: () => {
      current += 1;
    },
    isCurrent: (token) => token === current,
  };
}

/** Fit quality of one output slice, labelled for messages. */
export interface SliceFit extends PceDiagnostics {
  label: string;
}

export interface OrderInfo {
  requested: number;
  effective: number;
  nRows: number;
}

function fixed(v: number): string {
  return v.toFixed(2);
}

/**
 * The warnings Python's `jaxgsa.pce.analyze` raises about the fit, in plain
 * words and with the same thresholds:
 *
 * - order reduced to fit the sample budget,
 * - `explained_variance` below 0.5 on the worst slice,
 * - `cv_rmse / std(Y)` above sqrt(0.5) on the worst slice, unless the
 *   in-sample fit already failed (that is underfitting, not overfitting).
 *
 * Slices whose numbers are not finite are skipped, as in Python.
 */
export function pceFitWarnings(
  fits: SliceFit[],
  order: OrderInfo,
  fitRatio = 0.5,
): string[] {
  const out: string[] = [];

  if (order.effective < order.requested) {
    out.push(
      `The PCE order was lowered from ${order.requested} to ${order.effective} so the number of terms fits ` +
        `the sample budget (fit_ratio ${fitRatio}, ${order.nRows} rows). Add runs to use a higher order.`,
    );
  }

  const explained = fits.filter((f) => Number.isFinite(f.explainedVariance));
  let poorFit = false;

  if (explained.length > 0) {
    const worst = explained.reduce((a, b) => (b.explainedVariance < a.explainedVariance ? b : a));

    if (worst.explainedVariance < POORFIT_THRESHOLD) {
      poorFit = true;
      out.push(
        `Explained variance is ${fixed(worst.explainedVariance)} on ${worst.label}, below ${POORFIT_THRESHOLD}. ` +
          "The surrogate leaves more than half of the output variance unexplained. " +
          "Every index is computed from the fit, so the indices describe the surrogate and not your model. " +
          "Raise the order until the leave-one-out error stops falling.",
      );
    }
  }

  const ratios = fits.flatMap((f) => {
    const ratio = f.looRmse / f.stdY;

    return Number.isFinite(ratio) ? [{ f, ratio }] : [];
  });

  if (ratios.length > 0 && !poorFit) {
    const worst = ratios.reduce((a, b) => (b.ratio > a.ratio ? b : a));

    if (worst.ratio > LOO_RATIO_THRESHOLD) {
      out.push(
        `The leave-one-out error is ${fixed(worst.ratio)} times the spread of the output (std Y) on ${worst.f.label}. ` +
          "The surrogate predicts less than half of the output variance for runs it has not seen. " +
          "A high in-sample fit does not show this overfit. Use a lower order or add runs.",
      );
    }
  }

  return out;
}

/**
 * eFAST accuracy hint. Python only rejects a curve shorter than the minimum;
 * it does not warn about a short one. The studio adds this hint on its own.
 * On Ishigami (D = 3), S1 of x2 over 6 seeds (analytic 0.442) ranged from
 * 0.31 to 0.81 at 129 points per curve. At 257 and 513 points every seed was
 * within 0.002. The error at short curves depends on the seed.
 */
export const EFAST_LOW_N = 257;

export function efastLowNWarning(nPerCurve: number): string | null {
  if (!Number.isFinite(nPerCurve) || nPerCurve >= EFAST_LOW_N) return null;

  return (
    `This eFAST design has ${nPerCurve} points per curve. Short curves can give large errors that change with the seed. ` +
    "On the Ishigami test function, S1 of x2 ranged from 0.31 to 0.81 over 6 seeds at 129 points (analytic value 0.44). " +
    `At ${EFAST_LOW_N} points and above, every seed was within 0.002. ` +
    `Use at least ${EFAST_LOW_N} points per curve, or re-run with another seed to check the result. ` +
    "This is a studio hint, not a Python rule."
  );
}
