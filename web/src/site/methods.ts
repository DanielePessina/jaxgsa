/**
 * Method metadata for the UI: two-sentence descriptions (grounded in the
 * jaxgsa README method table) plus the small classified config surface each
 * method exposes.
 */

import type { DesignMethod, GivenDataMethod } from "./engine";

/** The methods exposed in the UI (kucherenko is not ported; see PLAN-WASM). */
export type MethodKey = "sobol" | "morris" | "pce" | "shapley";

export interface MethodMeta {
  name: string;
  tag: string;
  blurb: string;
}

export const DESIGN_METHOD_META: Record<DesignMethod, MethodMeta> = {
  sobol: {
    name: "sobol",
    tag: "Saltelli design · S1, ST",
    blurb:
      "The reference variance decomposition — first-order (S1) and total (ST) indices from a Saltelli design. " +
      "Use it when you can still choose where to run the model.",
  },
  kucherenko: {
    name: "kucherenko",
    tag: "conditional-copula design · S1, ST",
    blurb:
      "Variance decomposition (S1, ST) on a purpose-built conditional-copula design — the dedicated-design alternative to Saltelli. " +
      "It targets correlated inputs; correlation sampling isn't ported yet, and for independent inputs the Python package warns that sobol reaches the same indices in fewer runs.",
  },
  morris: {
    name: "morris",
    tag: "trajectory screening · mu, mu*, sigma",
    blurb:
      "Cheap trajectory-based screening that ranks inputs by elementary-effect means (mu, mu*) and spreads (sigma). " +
      "Use it on a tight budget to drop dead inputs before a full Sobol' run.",
  },
};

export const GIVEN_METHOD_META: Record<GivenDataMethod, MethodMeta> = {
  pce: {
    name: "pce",
    tag: "polynomial chaos · S1, ST",
    blurb:
      "Fits a polynomial-chaos surrogate to any (X, Y) data and reads S1/ST off its coefficients — the given-data route to Sobol indices. " +
      "Best for smooth models; fewest samples per unit of accuracy.",
  },
  shapley: {
    name: "shapley",
    tag: "PCE-based · Sh",
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