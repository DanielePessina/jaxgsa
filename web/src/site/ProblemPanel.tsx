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
import { demoById, DEMOS, type DemoId } from "./demos";
import { MonoSelect } from "./primitives";
import type { ProblemSpec } from "@/jaxgsa/sampling";

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

function blankRow(): ProblemRow {
  return {
    name: "",
    dist: "uniform",
    low: "",
    high: "",
    mean: "",
    variance: "",
    gLow: "",
    gHigh: "",
  };
}

function rowToSpec(row: ProblemRow): {
  spec: ProblemSpec["marginals"][number] | null;
  error: string | null;
} {
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
}: {
  problem: ProblemSpec | null;
  onProblemChange: (p: ProblemSpec) => void;
}) {
  const [rows, setRows] = useState<ProblemRow[]>(() =>
    rowsFromProblem(demoById("ishigami").problem),
  );
  const [demoId, setDemoId] = useState<DemoId>("ishigami");

  const specs = rows.map(rowToSpec);
  const names = rows.map((r) => r.name.trim());
  const dupNames = names.some((n, i) => n !== "" && names.indexOf(n) !== i);

  const validSpec =
    specs.every((s) => s.error === null) &&
    !dupNames &&
    rows.length >= 1
      ? {
          names: rows.map((r) => r.name.trim()),
          marginals: specs.map((s) => s.spec) as ProblemSpec["marginals"],
        }
      : null;

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
    setRows(rowsFromProblem(demoById(demoId).problem));
  };

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
                          onValueChange={(v) => set(i, { dist: v as DistKind })}
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
              onChange={(v) => setDemoId(v as DemoId)}
            >
              {DEMOS.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label} · {d.problem.names.length} inputs
                </option>
              ))}
            </MonoSelect>
            <Button type="button" variant="ghost" size="sm" onClick={loadDemo}>
              Load
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Demo problems include a built-in model and analytical reference
            indices, so you can try every method end to end before switching
            to your own problem.
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