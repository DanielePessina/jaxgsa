import rawKucherenko from "../../goldens/v0.9.0/kucherenko.json";
import rawSobol from "../../goldens/v0.9.0/sobol.json";

export interface Golden {
  x: number[][];
  y: number[];
  expected: { S1: number[]; ST: number[]; variance: number[] };
  tolerance: { rtol: number; atol: number };
}

interface RawGolden {
  method: string;
  n: number;
  x: number[][];
  y: number[];
  expected: { S1: number[]; ST: number[]; variance: number | number[] };
  tolerance: { rtol: number; atol: number };
}

const RAW = rawKucherenko as RawGolden;

export function loadKucherenkoGolden(): Golden {
  const variance = RAW.expected.variance;
  return {
    x: RAW.x,
    y: RAW.y,
    expected: {
      S1: RAW.expected.S1,
      ST: RAW.expected.ST,
      // The v0.9.0 fixture stores the single variance as a bare float, not a
      // one-element list; normalize it so consumers always see number[].
      variance: Array.isArray(variance) ? variance : [variance],
    },
    tolerance: RAW.tolerance,
  };
}

export interface SobolGolden {
  x: number[][];
  y: number[];
  expected: { S1: number[]; ST: number[] };
  tolerance: { rtol: number; atol: number };
  config: { baseN: number; calcSecondOrder: boolean };
}

interface RawSobolGolden {
  method: string;
  n: number;
  x: number[][];
  y: number[];
  expected: { S1: number[]; ST: number[] };
  tolerance: { rtol: number; atol: number };
  config: { base_n: number; calc_second_order: boolean };
}

const RAW_SOBOL = rawSobol as RawSobolGolden;

export function loadSobolGolden(): SobolGolden {
  return {
    x: RAW_SOBOL.x,
    y: RAW_SOBOL.y,
    expected: RAW_SOBOL.expected,
    tolerance: RAW_SOBOL.tolerance,
    config: {
      baseN: RAW_SOBOL.config.base_n,
      calcSecondOrder: RAW_SOBOL.config.calc_second_order,
    },
  };
}
