import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatIndex } from "./format";
import type { AnalysisResult } from "./engine";

/** Compact per-slice index table with plain-div horizontal bars. */
export function ResultsTable({
  result,
  sliceIndex,
}: {
  result: AnalysisResult;
  sliceIndex: number;
}) {
  const slice = result.slices[sliceIndex];

  if (!slice) return null;

  const maxAbs = slice.columns.map((c) => {
    let m = 0;

    for (const v of c.values) m = Math.max(m, Math.abs(v));

    return m;
  });

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-24">parameter</TableHead>
          {slice.columns.map((c) => (
            <TableHead key={c.key}>{c.label}</TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {result.parameters.map((name, i) => (
          <TableRow key={name}>
            <TableCell className="font-mono">{name}</TableCell>
            {slice.columns.map((c, j) => {
              const v = c.values[i];
              const pct = maxAbs[j] > 0 ? (Math.abs(v) / maxAbs[j]) * 100 : 0;
              const negative = v < 0;

              return (
                <TableCell key={c.key}>
                  <div className="font-mono text-sm tabular-nums">
                    {formatIndex(v)}
                  </div>
                  <div className="mt-1.5 h-1 w-full max-w-36 overflow-hidden rounded-sm bg-muted">
                    <div
                      className={`h-full rounded-sm ${negative ? "bg-destructive" : "bg-primary"}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </TableCell>
              );
            })}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}