import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { AnalysisResult } from "./engine";

function formatValue(v: number): string {
  return Number.isNaN(v) ? "NaN" : v.toFixed(4);
}

/**
 * Per-parameter index table with plain-div horizontal bars. Bar length is
 * |value| scaled to the column's max |value|; positive values use the accent,
 * negative values the destructive hue.
 */
export function ResultsTable({ result }: { result: AnalysisResult }) {
  const maxAbs = result.columns.map((c) => {
    let m = 0;
    for (const v of c.values) m = Math.max(m, Math.abs(v));
    return m;
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
          {result.method} — indices
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-24">parameter</TableHead>
              {result.columns.map((c) => (
                <TableHead key={c.key}>{c.label}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {result.parameters.map((name, i) => (
              <TableRow key={name}>
                <TableCell className="font-mono">{name}</TableCell>
                {result.columns.map((c, j) => {
                  const v = c.values[i];
                  const pct = maxAbs[j] > 0
                    ? (Math.abs(v) / maxAbs[j]) * 100
                    : 0;
                  const negative = v < 0;
                  return (
                    <TableCell key={c.key}>
                      <div className="font-mono text-sm tabular-nums">
                        {formatValue(v)}
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
        {result.notes.length > 0 && (
          <p className="text-xs text-muted-foreground">
            {result.notes.join(" ")}
          </p>
        )}
      </CardContent>
    </Card>
  );
}