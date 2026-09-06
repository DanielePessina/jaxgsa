import rawKucherenko from "../../goldens/v0.9.0/kucherenko.json";

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
