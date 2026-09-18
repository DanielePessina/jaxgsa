/**
 * Method metadata for the UI: two-sentence descriptions (grounded in the
 * jaxgsa README method table) plus the small classified config surface each
 * method exposes.
 */

import type { DesignMethod, GivenDataMethod } from "./engine";

/** Methods exposed in the UI. Kucherenko compute exists but its correlated
 * sampling route is not exposed yet, so it is intentionally absent here. */
export type MethodKey = "sobol" | "morris" | "pce" | "shapley";

export interface MethodMeta {
  name: string;
  tag: string;
  blurb: string;
  /** The X layout this estimator is intended to consume in the workbench. */
  input: { kind: "dedicated"; design: DesignMethod } | { kind: "given" };
}

export const DESIGN_METHOD_META: Record<DesignMethod, MethodMeta> = {
  sobol: {
    name: "sobol",
    tag: "Saltelli design · S1, ST",
    input: { kind: "dedicated", design: "sobol" },
    blurb:
      "The reference variance decomposition — first-order (S1) and total (ST) indices from a Saltelli design. " +
      "Use it when you can still choose where to run the model.",
  },
  kucherenko: {
    name: "kucherenko",
    tag: "conditional-copula design · S1, ST",
    input: { kind: "dedicated", design: "kucherenko" },
    blurb:
      "Variance decomposition (S1, ST) on a purpose-built conditional-copula design — the dedicated-design alternative to Saltelli. " +
      "It targets correlated inputs; correlation sampling isn't ported yet, and for independent inputs the Python package warns that sobol reaches the same indices in fewer runs.",
  },
  morris: {
    name: "morris",
    tag: "trajectory screening · mu, mu*, sigma",
    input: { kind: "dedicated", design: "morris" },
    blurb:
      "Cheap trajectory-based screening that ranks inputs by elementary-effect means (mu, mu*) and spreads (sigma). " +
      "Use it on a tight budget to drop dead inputs before a full Sobol' run.",
  },
};

export const GIVEN_METHOD_META: Record<GivenDataMethod, MethodMeta> = {
  pce: {
    name: "pce",
    tag: "polynomial chaos · S1, ST",
    input: { kind: "given" },
    blurb:
      "Fits a polynomial-chaos surrogate to any (X, Y) data and reads S1/ST off its coefficients — the given-data route to Sobol indices. " +
      "Best for smooth models; fewest samples per unit of accuracy.",
  },
  shapley: {
    name: "shapley",
    tag: "PCE-based · Sh",
    input: { kind: "given" },
    blurb:
      "One number per input that sums to exactly 1. " +
      "Computed from a PCE fit, with no permutation Monte Carlo.",
  },
};

export const METHOD_META: Record<MethodKey, MethodMeta> = {
  ...DESIGN_METHOD_META,
  ...GIVEN_METHOD_META,
};

export const ALL_METHODS: MethodKey[] = [
  "sobol",
  "morris",
  "pce",
  "shapley",
];
