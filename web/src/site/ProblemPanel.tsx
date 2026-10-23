import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { demoById, DEMOS, type Demo, type DemoId } from "./demos";
import {
  blankRow,
  type DistKind,
  type ProblemRow,
  type ProblemValidation,
} from "./problemRows";
import { MonoSelect } from "./primitives";
import type { ProblemSpec } from "@/jaxgsa/sampling";

export function ProblemPanel({
  problem,
  rows,
  onRowsChange,
  outputNamesRaw,
  onOutputNamesChange,
  demoId,
  onDemoIdChange,
  validation,
  onLoadDemo,
}: {
  /** The problem the rest of the app is using right now. */
  problem: ProblemSpec | null;
  rows: ProblemRow[];
  onRowsChange: (rows: ProblemRow[]) => void;
  outputNamesRaw: string;
  onOutputNamesChange: (raw: string) => void;
  /** The demo picked in the dropdown. It is not loaded until the user clicks. */
  demoId: DemoId;
  onDemoIdChange: (id: DemoId) => void;
  validation: ProblemValidation;
  onLoadDemo: (demo: Demo) => void;
}) {
  const { rowErrors, outputNamesError } = validation;

  const set = (i: number, patch: Partial<ProblemRow>) => {
    onRowsChange(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)));
  };

  const current = validation.problem ?? problem;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
            problem — model inputs
          </CardTitle>
          <CardDescription>
            One row per input of your model, with its distribution. The
            available distributions are uniform and (optionally truncated)
            gaussian; correlation and categorical marginals are not ported
            yet.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-border max-sm:sr-only">
                  <th scope="col" className="px-2 pb-2 text-left font-mono text-xs uppercase tracking-widest text-muted-foreground">
                    name
                  </th>
                  <th scope="col" className="px-2 pb-2 text-left font-mono text-xs uppercase tracking-widest text-muted-foreground">
                    distribution
                  </th>
                  <th scope="col" className="px-2 pb-2 text-left font-mono text-xs uppercase tracking-widest text-muted-foreground">
                    parameters
                  </th>
                  <th scope="col" className="w-10"><span className="sr-only">remove</span></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => {
                  const err = rowErrors[i];

                  return (
                    <tr key={i} className="border-b border-border/60 max-sm:flex max-sm:flex-wrap max-sm:items-center">
                      <td className="px-2 py-2">
                        <Input
                          value={row.name}
                          aria-label={`Name of input ${i + 1}`}
                          placeholder="x1"
                          onChange={(e) => set(i, { name: e.target.value })}
                          className="w-24 font-mono"
                        />
                      </td>
                      <td className="px-2 py-2">
                        <Select
                          value={row.dist}
                          onValueChange={(v) => {
                            // SAFETY: Select only emits the two values declared below.
                            set(i, { dist: v as DistKind });
                          }}
                        >
                          <SelectTrigger aria-label={`Distribution of input ${i + 1}`} className="w-32 font-mono text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="uniform" className="font-mono">
                              uniform
                            </SelectItem>
                            <SelectItem value="gaussian" className="font-mono">
                              gaussian
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </td>
                      <td className="px-2 py-2 max-sm:order-last max-sm:basis-full">
                        {row.dist === "uniform" ? (
                          <div className="flex flex-wrap items-center gap-1.5">
                            <Input
                              value={row.low}
                            aria-label={`Lower bound of input ${i + 1}`}
                              placeholder="low"
                              onChange={(e) => set(i, { low: e.target.value })}
                              className="w-24 font-mono"
                            />
                            <span className="font-mono text-xs text-muted-foreground">to</span>
                            <Input
                              value={row.high}
                            aria-label={`Upper bound of input ${i + 1}`}
                              placeholder="high"
                              onChange={(e) => set(i, { high: e.target.value })}
                              className="w-24 font-mono"
                            />
                          </div>
                        ) : (
                          <div className="flex flex-wrap items-center gap-1.5">
                            <Input
                              value={row.mean}
                            aria-label={`Mean of input ${i + 1}`}
                              placeholder="mean"
                              onChange={(e) => set(i, { mean: e.target.value })}
                              className="w-20 font-mono"
                            />
                            <Input
                              value={row.variance}
                            aria-label={`Variance of input ${i + 1}`}
                              placeholder="variance"
                              onChange={(e) => set(i, { variance: e.target.value })}
                              className="w-24 font-mono"
                            />
                            <Input
                              value={row.gLow}
                            aria-label={`Truncation lower bound (optional) of input ${i + 1}`}
                              placeholder="low (optional)"
                              onChange={(e) => set(i, { gLow: e.target.value })}
                              className="w-28 font-mono"
                            />
                            <Input
                              value={row.gHigh}
                            aria-label={`Truncation upper bound (optional) of input ${i + 1}`}
                              placeholder="high (optional)"
                              onChange={(e) => set(i, { gHigh: e.target.value })}
                              className="w-28 font-mono"
                            />
                          </div>
                        )}
                        {err && (
                          <p className="mt-1 text-xs text-destructive">{err}</p>
                        )}
                      </td>
                      <td className="px-2 py-2">
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          disabled={rows.length <= 1}
                          onClick={() => onRowsChange(rows.filter((_, j) => j !== i))}
                          className="font-mono text-xs"
                        >
                          Remove
                        </Button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="space-y-1.5">
            <label htmlFor="output-names" className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
              output names
            </label>
            <div className="flex flex-wrap items-center gap-2">
              <Input
                id="output-names"
                value={outputNamesRaw}
                placeholder="y (default)"
                onChange={(e) => onOutputNamesChange(e.target.value)}
                className="w-72 font-mono"
              />
              <span className="text-xs text-muted-foreground">
                comma-separated; shown as the expected Y column names in the schema guide. The headers of your uploaded file name the results.
              </span>
            </div>
            {outputNamesError && (
              <p className="text-xs text-destructive">{outputNamesError}</p>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => onRowsChange([...rows, blankRow()])}
            >
              + Add parameter
            </Button>
            <span className="mx-1 h-4 w-px bg-border" />
            <span className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
              load demo problem
            </span>
            <MonoSelect
              aria-label="Demo problem"
              value={demoId}
              onChange={(v) => {
                // SAFETY: MonoSelect options are generated exclusively from DEMOS.
                onDemoIdChange(v as DemoId);
              }}
            >
              {DEMOS.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label} · {d.problem.names.length} inputs
                </option>
              ))}
            </MonoSelect>
            <Button type="button" variant="outline" size="sm" onClick={() => onLoadDemo(demoById(demoId))}>
              Start Sobol demo
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            This loads the benchmark, creates its Saltelli design, evaluates
            the built-in model, and prepares a Sobol analysis. The benchmark
            itself is not tied to one method; other routes remain available.
          </p>

          {current && (
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline" className="font-mono text-xs">
                D = {current.names.length}
              </Badge>
              {current.names.map((n, j) => {
                const m = current.marginals[j];

                const label =
                  m.kind === "uniform"
                    ? `~ U(${m.low.toFixed(3)}, ${m.high.toFixed(3)})`
                    : m.kind === "gaussian"
                      ? `~ N(${m.mean}, ${m.variance}${m.low !== undefined || m.high !== undefined ? " truncated" : ""})`
                      : "categorical";

                return (
                  <span
                    key={n}
                    className="rounded-sm border border-border bg-muted/40 px-2 py-0.5 font-mono text-xs text-muted-foreground"
                  >
                    {n} {label}
                  </span>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
