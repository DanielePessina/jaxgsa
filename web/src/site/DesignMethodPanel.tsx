import { useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { CsvUpload } from "./CsvUpload";
import { ResultsTable } from "./ResultsTable";
import {
  ISHIGAMI_PROBLEM,
  alignYByRunId,
  analyzeGenerated,
  buildDesignCsv,
  evaluateIshigami,
  generateDesign,
  parseCsv,
  type AnalysisResult,
  type DesignMethod,
  type GeneratedDesign,
} from "./engine";

const METHOD_BLURB: Record<DesignMethod, string> = {
  sobol: "Saltelli design. First/total-order indices (S2 estimator not ported yet).",
  morris: "Trajectory design. Elementary effects mu, mu*, sigma.",
  kucherenko: "Conditional-copula design (Saltelli column swap for independent inputs). S1, ST.",
};

interface ConfigState {
  baseN: string;
  secondOrder: boolean;
  seed: string;
  nTrajectories: string;
  numLevels: string;
  nSamples: string;
}

const DEFAULT_CONFIG: ConfigState = {
  baseN: "128",
  secondOrder: false,
  seed: "0",
  nTrajectories: "20",
  numLevels: "4",
  nSamples: "100",
};

export function DesignMethodPanel({ method }: { method: DesignMethod }) {
  const [cfg, setCfg] = useState<ConfigState>(DEFAULT_CONFIG);
  const [design, setDesign] = useState<GeneratedDesign | null>(null);
  const [y, setY] = useState<Float64Array | null>(null);
  const [yLabel, setYLabel] = useState<string>("no outputs loaded");
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const set = <K extends keyof ConfigState>(key: K, value: ConfigState[K]) =>
    setCfg((c) => ({ ...c, [key]: value }));

  const invalidate = () => {
    setY(null);
    setYLabel("no outputs loaded");
    setResults(null);
  };

  const onGenerate = () => {
    setError(null);
    try {
      const config =
        method === "sobol"
          ? { baseN: Number(cfg.baseN), calcSecondOrder: cfg.secondOrder, seed: Number(cfg.seed) }
          : method === "morris"
            ? {
                nTrajectories: Number(cfg.nTrajectories),
                numLevels: Number(cfg.numLevels),
                seed: Number(cfg.seed),
              }
            : { nSamples: Number(cfg.nSamples), seed: Number(cfg.seed) };
      const gen = generateDesign(method, ISHIGAMI_PROBLEM, config);
      setDesign(gen);
      invalidate();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const onEvaluate = () => {
    if (!design) return;
    setError(null);
    setY(evaluateIshigami(design));
    setYLabel(`Ishigami example model · ${design.nRuns} runs (in-browser)`);
    setResults(null);
  };

  const onUploadY = (parsed: ReturnType<typeof parseCsv>) => {
    if (!design) return;
    setError(null);
    try {
      const idCol = parsed.headers.findIndex((h) => h.trim() === "run_id");
      if (idCol < 0) {
        throw new Error(
          'the Y CSV must contain a "run_id" column matching the design download',
        );
      }
      const ids = parsed.rows.map((r) => r[idCol]);
      const aligned = alignYByRunId(parsed.rows, ids, design);
      setY(aligned);
      setYLabel(`uploaded Y CSV · ${aligned.length} runs`);
      setResults(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const onRun = async () => {
    if (!design || !y) return;
    setError(null);
    setBusy(true);
    await new Promise((r) => setTimeout(r, 10));
    try {
      setResults(analyzeGenerated(design, y));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const onDownload = () => {
    if (!design) return;
    const csv = buildDesignCsv(design.problem, design);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `jaxgsa-${method}-design.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const preview = design
    ? Array.from({ length: Math.min(10, design.nRuns) }, (_, r) => {
        const row = new Array<number>(design.nParams);
        for (let j = 0; j < design.nParams; j++) {
          row[j] = design.samples[r * design.nParams + j];
        }
        return { id: r, row };
      })
    : [];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
            problem (read-only demo)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-foreground">
            <span className="font-mono">Ishigami</span> — f(x) = sin(x₁) + 7
            sin²(x₂) + 0.1 x₃⁴ sin(x₁)
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            {ISHIGAMI_PROBLEM.names.map((n) => (
              <span
                key={n}
                className="rounded-sm border border-border bg-muted/40 px-2 py-0.5 font-mono text-xs text-muted-foreground"
              >
                {n} ~ Uniform(−π, π)
              </span>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
            {method} — configuration
          </CardTitle>
          <CardDescription>{METHOD_BLURB[method]}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            {method === "sobol" && (
              <>
                <div className="space-y-1.5">
                  <Label htmlFor="baseN" className="font-mono text-xs">
                    base_n
                  </Label>
                  <Input
                    id="baseN"
                    type="number"
                    min={16}
                    step={1}
                    value={cfg.baseN}
                    onChange={(e) => set("baseN", e.target.value)}
                    className="font-mono"
                  />
                </div>
                <div className="flex items-end space-x-2 pb-2">
                  <input
                    id="secondOrder"
                    type="checkbox"
                    checked={cfg.secondOrder}
                    onChange={(e) => set("secondOrder", e.target.checked)}
                    className="size-4 accent-[oklch(0.78_0.14_195)]"
                  />
                  <Label htmlFor="secondOrder" className="text-xs">
                    calc_second_order
                  </Label>
                </div>
              </>
            )}
            {method === "morris" && (
              <>
                <div className="space-y-1.5">
                  <Label htmlFor="nTraj" className="font-mono text-xs">
                    n_trajectories
                  </Label>
                  <Input
                    id="nTraj"
                    type="number"
                    min={2}
                    step={1}
                    value={cfg.nTrajectories}
                    onChange={(e) => set("nTrajectories", e.target.value)}
                    className="font-mono"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="numLevels" className="font-mono text-xs">
                    num_levels
                  </Label>
                  <Input
                    id="numLevels"
                    type="number"
                    min={2}
                    step={1}
                    value={cfg.numLevels}
                    onChange={(e) => set("numLevels", e.target.value)}
                    className="font-mono"
                  />
                </div>
              </>
            )}
            {method === "kucherenko" && (
              <div className="space-y-1.5">
                <Label htmlFor="nSamples" className="font-mono text-xs">
                  n_samples (per block)
                </Label>
                <Input
                  id="nSamples"
                  type="number"
                  min={2}
                  step={1}
                  value={cfg.nSamples}
                  onChange={(e) => set("nSamples", e.target.value)}
                  className="font-mono"
                />
              </div>
            )}
            <div className="space-y-1.5">
              <Label htmlFor="seed" className="font-mono text-xs">
                seed
              </Label>
              <Input
                id="seed"
                type="number"
                step={1}
                value={cfg.seed}
                onChange={(e) => set("seed", e.target.value)}
                className="font-mono"
              />
            </div>
          </div>
          <Button type="button" onClick={onGenerate}>
            Generate design
          </Button>
        </CardContent>
      </Card>

      {design && (
        <Card>
          <CardHeader>
            <CardTitle className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
              design — {design.nRuns} runs
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-x-6 gap-y-1">
              {design.summary.map(([k, v]) => (
                <span key={k} className="font-mono text-xs">
                  <span className="text-muted-foreground">{k}: </span>
                  <span className="tabular-nums">{v}</span>
                </span>
              ))}
            </div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-16 font-mono">run_id</TableHead>
                  {design.problem.names.map((n) => (
                    <TableHead key={n} className="font-mono">
                      {n}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {preview.map(({ id, row }) => (
                  <TableRow key={id}>
                    <TableCell className="font-mono text-muted-foreground">
                      {id}
                    </TableCell>
                    {row.map((v, j) => (
                      <TableCell key={j} className="font-mono tabular-nums">
                        {v.toFixed(6)}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <p className="text-xs text-muted-foreground">
              showing first {Math.min(10, design.nRuns)} of {design.nRuns} rows
              — download the full design to evaluate your model
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <Button type="button" variant="outline" size="sm" onClick={onDownload}>
                Download design CSV
              </Button>
              <Button type="button" size="sm" onClick={onEvaluate}>
                Evaluate example model (Ishigami)
              </Button>
            </div>
            <CsvUpload
              label="Upload outputs (Y) CSV"
              hint="CSV with a run_id column (any order) + one output column"
              disabled={busy}
              onLoaded={onUploadY}
              onError={setError}
            />
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="font-mono text-xs text-muted-foreground">{yLabel}</p>
              <Button
                type="button"
                disabled={!y || busy}
                onClick={() => void onRun()}
              >
                {busy ? "computing…" : "Run analysis"}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {results && <ResultsTable result={results} />}

      {error && (
        <Alert variant="destructive">
          <AlertTitle>engine error</AlertTitle>
          <AlertDescription className="font-mono">{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}