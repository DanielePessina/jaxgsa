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
      "Estimate each input's share of output variation alone (S1) and with interactions (ST). " +
      "Use it when you can evaluate your model at the inputs generated here.",
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
      "Screen many inputs with fewer model runs. " +
      "Mu-star ranks overall influence; sigma suggests interactions or nonlinear effects.",
  },
};

export const GIVEN_METHOD_META: Record<GivenDataMethod, MethodMeta> = {
  pce: {
    name: "pce",
    tag: "polynomial chaos · S1, ST",
    input: { kind: "given" },
    blurb:
      "Fit a polynomial approximation to model runs you already have, then estimate S1 and ST. " +
      "Best for smooth models; check the fit before trusting its indices.",
  },
  shapley: {
    name: "shapley",
    tag: "PCE-based · Sh",
    input: { kind: "given" },
    blurb:
      "Allocate the fitted model's output variation across inputs, including interactions. " +
      "The shares sum to one; check the PCE fit before interpreting them.",
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
