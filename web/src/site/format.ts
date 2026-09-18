/**
 * Small shared formatting helpers for index values (editorial: concise,
 * no trailing zeros).
 */
export function formatIndex(v: number): string {
  if (Number.isNaN(v)) return "NaN";

  if (v === 0) return "0";
  const s = Math.abs(v);

  if (s >= 1e-3) return String(Number(v.toFixed(4)));

  if (s < 1e-9) return "0";

  return v.toExponential(2);
}