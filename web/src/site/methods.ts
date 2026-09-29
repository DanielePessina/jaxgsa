/**
 * Method metadata for the UI: two-sentence descriptions (grounded in the
 * jaxgsa README method table) plus the small classified config surface each
 * method exposes.
 */

import type { DesignMethod, GivenDataMethod } from "./engine";

/** Methods exposed in the UI. All compute routes exist in the WASM ports;
 * kucherenko was added alongside the others. */
export type MethodKey =
  | "sobol"
  | "morris"
  | "kucherenko"
  | "efast"
  | "pce"
  | "shapley"
  | "borgonovo"
  | "pawn";

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
  efast: {
    name: "efast",
    tag: "Fourier design · S1, ST",
    input: { kind: "dedicated", design: "efast" },
    blurb:
      "Estimate S1 and ST from a deterministic sinusoidal search-curve design. " +
      "No bootstrap by design — rerun with different phases to gauge stability.",
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
  borgonovo: {
    name: "borgonovo",
    tag: "moment-independent · delta",
    input: { kind: "given" },
    blurb:
      "Measure how much fixing a parameter shifts the whole output distribution, " +
      "not just its variance. Useful when the output is skewed or multimodal.",
  },
  pawn: {
    name: "pawn",
    tag: "distribution screening · KS",
    input: { kind: "given" },
    blurb:
      "Compare each parameter's conditioned output distributions with the overall one " +
      "by the Kolmogorov-Smirnov distance. A screening complement to variance indices.",
  },
};

export const METHOD_META: Record<MethodKey, MethodMeta> = {
  ...DESIGN_METHOD_META,
  ...GIVEN_METHOD_META,
};

export const ALL_METHODS: MethodKey[] = [
  "sobol",
  "morris",
  "kucherenko",
  "efast",
  "pce",
  "shapley",
  "borgonovo",
  "pawn",
];
