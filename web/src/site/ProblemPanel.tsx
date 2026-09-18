import { useEffect, useState } from "react";
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
import { MonoSelect } from "./primitives";
import type { MarginalSpec, ProblemSpec } from "@/jaxgsa/sampling";

type DistKind = "uniform" | "gaussian";

interface ProblemRow {
  name: string;
  dist: DistKind;
  low: string;
  high: string;
  mean: string;
  variance: string;
  gLow: string;
  gHigh: string;
}

interface RowSpecResult {
  spec: MarginalSpec | null;
  error: string | null;
}

/**
 * Comma-separated optional output labels, normalized to a trimmed list.
 * Empty input yields `null` (the package default: outputs are just `y`).
 */
function outputNamesFromInput(raw: string): string[] | null {
  const names = raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s !== "");

  if (names.length === 0) return null;
  const dup = names.find((n, i) => names.indexOf(n) !== i);

  if (dup !== undefined) throw new Error(`duplicate output name: ${dup}`);

  return names;
}

function blankRow(): ProblemRow {
  return {
    name: "",
    dist: "uniform",
    low: "0",
    high: "1",
    mean: "",
    variance: "",
    gLow: "",
    gHigh: "",
  };
}

function rowToSpec(row: ProblemRow): RowSpecResult {
  const name = row.name.trim();

  if (name === "") return { spec: null, error: "name is required" };

  if (row.dist === "uniform") {
    const low = Number(row.low);
    const high = Number(row.high);

    if (row.low.trim() === "" || row.high.trim() === "" || Number.isNaN(low) || Number.isNaN(high)) {
      return { spec: null, error: "low and high must be numbers" };
    }

    if (low >= high) return { spec: null, error: "low must be < high" };

    return { spec: { kind: "uniform", low, high }, error: null };
  }

  const mean = Number(row.mean);
  const variance = Number(row.variance);

  if (row.mean.trim() === "" || row.variance.trim() === "" || Number.isNaN(mean) || Number.isNaN(variance)) {
    return { spec: null, error: "mean and variance must be numbers" };
  }

  if (variance <= 0) return { spec: null, error: "variance must be > 0" };
  const low = row.gLow.trim() === "" ? undefined : Number(row.gLow);
  const high = row.gHigh.trim() === "" ? undefined : Number(row.gHigh);

  if (low !== undefined && Number.isNaN(low)) return { spec: null, error: "lower bound must be a number" };

  if (high !== undefined && Number.isNaN(high)) return { spec: null, error: "upper bound must be a number" };

  if (low !== undefined && high !== undefined && low >= high) {
    return { spec: null, error: "lower bound must be < upper bound" };
  }

  return { spec: { kind: "gaussian", mean, variance, low, high }, error: null };
}

function rowsFromProblem(p: ProblemSpec): ProblemRow[] {
  return p.names.map((name, j) => {
    const m = p.marginals[j];

    if (m.kind === "uniform") {
      return {
        name,
        dist: "uniform" as const,
        low: String(m.low),
        high: String(m.high),
        mean: "",
        variance: "",
        gLow: "",
        gHigh: "",
      };
    }

    if (m.kind !== "gaussian") {
      throw new Error("categorical marginals are not ported yet");
    }

    return {
      name,
      dist: "gaussian" as const,
      low: "",
      high: "",
      mean: String(m.mean),
      variance: String(m.variance),
      gLow: m.low === undefined ? "" : String(m.low),
      gHigh: m.high === undefined ? "" : String(m.high),
    };
  });
}

export function ProblemPanel({
  problem,
  onProblemChange,
  onDemoChange,
  onLoadDemo,
}: {
  problem: ProblemSpec | null;
  onProblemChange: (p: ProblemSpec) => void;
  onDemoChange?: (demo: Demo) => void;
  onLoadDemo?: (demo: Demo) => void;
}) {
  const [rows, setRows] = useState<ProblemRow[]>(() =>
    rowsFromProblem(demoById("linear").problem),
  );

  const [demoId, setDemoId] = useState<DemoId>("linear");
  const [outputNamesRaw, setOutputNamesRaw] = useState("");

  const specs = rows.map(rowToSpec);
  const names = rows.map((r) => r.name.trim());
  const dupNames = names.some((n, i) => n !== "" && names.indexOf(n) !== i);

  let outputNames: string[] | null = null;
  let outputNamesError: string | null = null;

  if (outputNamesRaw.trim() !== "") {
    try {
      outputNames = outputNamesFromInput(outputNamesRaw);
    } catch (err) {
      outputNames = null;
      outputNamesError = err instanceof Error ? err.message : String(err);
    }
  }

  let validSpec: ProblemSpec | null = null;

  if (
    specs.every((s) => s.error === null) &&
    !dupNames &&
    rows.length >= 1 &&
    outputNamesError === null
  ) {
    const marginals: ProblemSpec["marginals"] = specs.map((s) => {
      if (s.spec === null) throw new Error("validated problem has a missing marginal");

      return s.spec;
    });

    validSpec = { names: rows.map((r) => r.name.trim()), marginals };

    if (outputNames) validSpec.outputNames = outputNames;
  }

  // `rows` only changes identity on a real edit (set/add/remove/load), so
  // this fires exactly when the problem definition changes — never on
  // unrelated re-renders from the parent.
  useEffect(() => {
    if (validSpec) onProblemChange(validSpec);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, onProblemChange]);

  const set = (i: number, patch: Partial<ProblemRow>) => {
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)));
  };

  const loadDemo = () => {
    const demo = demoById(demoId);
    setRows(rowsFromProblem(demo.problem));
    onLoadDemo?.(demo);
  };

  // Keep the app informed of the demo selected here so the Analyze section's
  // "Load Demo Results" tracks this tab, not just Ishigami.
  useEffect(() => {
    onDemoChange?.(demoById(demoId));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demoId]);

  const current = validSpec ?? problem;

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
                <tr className="border-b border-border">
                  <th className="px-2 pb-2 text-left font-mono text-xs uppercase tracking-widest text-muted-foreground">
                    name
                  </th>
                  <th className="px-2 pb-2 text-left font-mono text-xs uppercase tracking-widest text-muted-foreground">
                    distribution
                  </th>
                  <th className="px-2 pb-2 text-left font-mono text-xs uppercase tracking-widest text-muted-foreground">
                    parameters
                  </th>
                  <th className="w-10" />
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => {
                  const err = specs[i].error ?? (dupNames ? "names must be unique" : null);

                  return (
                    <tr key={i} className="border-b border-border/60">
                      <td className="px-2 py-2">
                        <Input
                          value={row.name}
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
                          <SelectTrigger className="w-32 font-mono text-xs">
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
                      <td className="px-2 py-2">
                        {row.dist === "uniform" ? (
                          <div className="flex items-center gap-1.5">
                            <Input
                              value={row.low}
                              placeholder="low"
                              onChange={(e) => set(i, { low: e.target.value })}
                              className="w-24 font-mono"
                            />
                            <span className="font-mono text-xs text-muted-foreground">to</span>
                            <Input
                              value={row.high}
                              placeholder="high"
                              onChange={(e) => set(i, { high: e.target.value })}
                              className="w-24 font-mono"
                            />
                          </div>
                        ) : (
                          <div className="flex flex-wrap items-center gap-1.5">
                            <Input
                              value={row.mean}
                              placeholder="mean"
                              onChange={(e) => set(i, { mean: e.target.value })}
                              className="w-20 font-mono"
                            />
                            <Input
                              value={row.variance}
                              placeholder="variance"
                              onChange={(e) => set(i, { variance: e.target.value })}
                              className="w-24 font-mono"
                            />
                            <Input
                              value={row.gLow}
                              placeholder="low (optional)"
                              onChange={(e) => set(i, { gLow: e.target.value })}
                              className="w-28 font-mono"
                            />
                            <Input
                              value={row.gHigh}
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
                          onClick={() =>
                            setRows((rs) => rs.filter((_, j) => j !== i))
                          }
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
                onChange={(e) => setOutputNamesRaw(e.target.value)}
                className="w-72 font-mono"
              />
              <span className="text-xs text-muted-foreground">
                comma-separated; used to name the Y columns you upload
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
              onClick={() => setRows((rs) => [...rs, blankRow()])}
            >
              + Add parameter
            </Button>
            <span className="mx-1 h-4 w-px bg-border" />
            <span className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
              load demo problem
            </span>
            <MonoSelect
              value={demoId}
              onChange={(v) => {
                // SAFETY: MonoSelect options are generated exclusively from DEMOS.
                setDemoId(v as DemoId);
              }}
            >
              {DEMOS.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label} · {d.problem.names.length} inputs
                </option>
              ))}
            </MonoSelect>
            <Button type="button" variant="outline" size="sm" onClick={loadDemo}>
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
