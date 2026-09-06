import { useEffect, useState } from "react";
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
import { CsvUpload } from "./CsvUpload";
import { demoForProblem } from "./demos";
import {
  alignYByRunId,
  analyzeCloud,
  analyzeGenerated,
  loadDemoCloud,
  parseXGiven,
  parseYGiven,
  type AnalysisResult,
  type DesignMethod,
  type GeneratedDesign,
  type ParsedCsv,
} from "./engine";
import { ALL_METHODS, METHOD_META, type MethodKey } from "./methods";
import { ErrorBanner, MonoSelect } from "./primitives";
import { useBusyAction } from "./hooks";
import type { ProblemSpec } from "@/jaxgsa/sampling";

type XSource = { kind: "design"; method: DesignMethod } | { kind: "uploaded" };

interface MethodAvailability {
  available: boolean;
  reason: string | null;
}

function availability(
  x: XSource | null,
): Record<MethodKey, MethodAvailability> {
  const design = x?.kind === "design" ? x.method : null;
  const uploaded = x?.kind === "uploaded";
  const needsDesign = (m: DesignMethod) =>
    design === m
      ? { available: true, reason: null }
      : {
          available: false,
          reason: uploaded
            ? `Its analysis needs the ${m} design; only PCE and Shapley run on arbitrary point clouds.`
            : `Requires the ${m} design — generate one in section 02.`,
        };
  return {
    sobol: needsDesign("sobol"),
    kucherenko: needsDesign("kucherenko"),
    morris: needsDesign("morris"),
    pce: { available: true, reason: null },
    shapley: { available: true, reason: null },
  };
}

export function AnalyzePanel({
  problem,
  designs,
  yData,
  setYData,
  appendResult,
  onGoSample,
  onGoResults,
}: {
  problem: ProblemSpec | null;
  designs: Partial<Record<DesignMethod, GeneratedDesign>>;
  yData: { values: Float64Array; label: string } | null;
  setYData: (y: { values: Float64Array; label: string } | null) => void;
  appendResult: (r: AnalysisResult) => void;
  onGoSample: () => void;
  onGoResults: () => void;
}) {
  const [xSource, setXSource] = useState<XSource | null>(null);
  const [givenX, setGivenX] = useState<Float64Array | null>(null);
  const [givenXLabel, setGivenXLabel] = useState("no X data loaded");
  const [order, setOrder] = useState("3");
  const [selected, setSelected] = useState<Record<MethodKey, boolean>>({
    sobol: false,
    kucherenko: false,
    morris: false,
    pce: true,
    shapley: false,
  });
  const { busy, error, setError, run } = useBusyAction();

  const demo = demoForProblem(problem);
  const avail = availability(xSource);
  const designMethods = Object.keys(designs) as DesignMethod[];

  useEffect(() => {
    if (xSource?.kind === "design" && !designs[xSource.method]) {
      const methods = Object.keys(designs) as DesignMethod[];
      const last = methods[methods.length - 1];
      setXSource(last ? { kind: "design", method: last } : null);
      setYData(null);
    }
  }, [designs, xSource, setYData]);

  const pickDesign = (method: DesignMethod) => {
    setXSource({ kind: "design", method });
    setGivenX(null);
    setYData(null);
    pruneDesignSelection();
    setError(null);
  };

  const pickUploaded = (x: Float64Array, label: string) => {
    setXSource({ kind: "uploaded" });
    setGivenX(x);
    setGivenXLabel(label);
    setYData(null);
    pruneDesignSelection();
    setError(null);
  };

  /** Design-locked methods only apply to their own design; uncheck them on X change. */
  const pruneDesignSelection = () => {
    setSelected((s) => ({
      ...s,
      sobol: false,
      kucherenko: false,
      morris: false,
    }));
  };

  const onUploadX = (parsed: ParsedCsv) => {
    if (!problem) return;
    try {
      const flat = parseXGiven(parsed, problem);
      pickUploaded(
        flat,
        `uploaded X · ${flat.length / problem.names.length} rows × ${problem.names.length} cols`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const onDemoCloud = () => {
    if (!demo) return;
    const { x, y, n } = loadDemoCloud(demo, 1024);
    pickUploaded(x, `demo cloud (${demo.label}) · ${n} rows`);
    setYData({ values: y, label: `demo model (${demo.label}) · ${n} outputs` });
  };

  const onUploadY = (parsed: ParsedCsv) => {
    if (!problem) return;
    setError(null);
    try {
      if (xSource?.kind === "design") {
        const design = designs[xSource.method];
        if (!design) return;
        const idCol = parsed.headers.findIndex((h) => h.trim() === "run_id");
        if (idCol < 0) {
          throw new Error(
            'the Y CSV must contain a "run_id" column matching the design download',
          );
        }
        const ids = parsed.rows.map((r) => r[idCol]);
        const aligned = alignYByRunId(parsed.rows, ids, design);
        setYData({ values: aligned, label: `uploaded Y · ${aligned.length} runs` });
      } else {
        const flat = parseYGiven(parsed);
        const N = givenX ? givenX.length / problem.names.length : 0;
        if (flat.length !== N) {
          throw new Error(`X has ${N} rows but Y has ${flat.length} values`);
        }
        setYData({ values: flat, label: `uploaded Y · ${flat.length} outputs` });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const onDemoY = () => {
    if (!demo || xSource?.kind !== "design") return;
    const design = designs[xSource.method];
    if (!design) return;
    setError(null);
    const y = demo.evaluate(design);
    setYData({ values: y, label: `demo model (${demo.label}) · ${y.length} runs` });
  };

  const selectedAvailable = ALL_METHODS.filter(
    (m) => selected[m] && avail[m].available,
  );

  const onRun = async () => {
    if (!problem || !yData) return;
    if (selectedAvailable.length === 0) {
      setError("Select at least one available method.");
      return;
    }
    const design = xSource?.kind === "design" ? designs[xSource.method] : null;
    const x = design ? design.samples : givenX;
    if (!x) return;
    const ok = await run(() => {
      for (const m of selectedAvailable) {
        if (m === "pce" || m === "shapley") {
          appendResult(analyzeCloud(m, problem, x, yData.values, Number(order)));
        } else if (design) {
          appendResult(analyzeGenerated(design, yData.values));
        }
      }
    });
    if (ok) onGoResults();
  };

  if (!problem) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          Define a problem in section 01 first.
        </CardContent>
      </Card>
    );
  }

  const xLabel =
    xSource?.kind === "design"
      ? `X = ${xSource.method} design · ${designs[xSource.method]?.nRuns} runs (sampled in section 02)`
      : givenXLabel;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
            input data — X and Y
          </CardTitle>
          <CardDescription>
            X is either a design you sampled above or your own point cloud.
            The methods you can run depend on which one you load — PCE and
            Shapley work on any data, the other three only on their own
            design.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
              X source
            </span>
            <MonoSelect
              value={
                xSource === null
                  ? "none"
                  : xSource.kind === "design"
                    ? xSource.method
                    : "uploaded"
              }
              onChange={(v) => {
                if (v === "uploaded") {
                  if (givenX) {
                    setXSource({ kind: "uploaded" });
                    setYData(null);
                  } else {
                    setError(
                      "Upload an X CSV (or load a demo cloud) to use your own data.",
                    );
                  }
                } else if (designs[v as DesignMethod]) {
                  pickDesign(v as DesignMethod);
                }
              }}
            >
              <option value="none" disabled>
                no X loaded
              </option>
              {designMethods.map((m) => (
                <option key={m} value={m}>
                  {m} design · {designs[m]?.nRuns} runs
                </option>
              ))}
              <option value="uploaded">my own X</option>
            </MonoSelect>
            <CsvUpload
              label="Upload X CSV"
              hint={`header: ${problem.names.join(", ")} — N rows`}
              disabled={busy}
              onLoaded={onUploadX}
              onError={setError}
              exampleLabel={demo ? `load demo cloud (${demo.label})` : undefined}
              onExample={demo ? onDemoCloud : undefined}
              exampleDisabled={busy}
            />
          </div>

          <p className="font-mono text-xs text-muted-foreground">{xLabel}</p>

          {designMethods.length === 0 && (
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-xs text-muted-foreground">
                No designs sampled yet — generate one in section 02, or upload
                your own X above.
              </p>
              <Button type="button" variant="outline" size="sm" onClick={onGoSample}>
                Sample a design
              </Button>
            </div>
          )}

          <div className="border-t border-border pt-4">
            <CsvUpload
              label="Upload outputs (Y) CSV"
              hint={
                xSource?.kind === "design"
                  ? "CSV with a run_id column (any order) + one output column"
                  : "one output column, one value per X row"
              }
              disabled={busy}
              onLoaded={onUploadY}
              onError={setError}
              exampleLabel={
                demo && xSource?.kind === "design"
                  ? `evaluate demo model (${demo.label})`
                  : undefined
              }
              onExample={demo && xSource?.kind === "design" ? onDemoY : undefined}
              exampleDisabled={busy}
            />
            <p className="mt-2 font-mono text-xs text-muted-foreground">
              {yData?.label ?? "no outputs loaded"}
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
            methods
          </CardTitle>
          <CardDescription>
            Select the analyses to run on this data. Methods that need a
            different design are listed dimmed with the reason.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-4">
            <div className="space-y-1.5">
              <Label htmlFor="order" className="font-mono text-xs">
                pce order (pce · shapley)
              </Label>
              <Input
                id="order"
                type="number"
                min={1}
                step={1}
                value={order}
                onChange={(e) => setOrder(e.target.value)}
                className="w-28 font-mono"
              />
            </div>
          </div>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {ALL_METHODS.map((m) => {
              const a = avail[m];
              const on = selected[m];
              return (
                <label
                  key={m}
                  className={`rounded-sm border p-3 transition-colors ${
                    !a.available
                      ? "cursor-not-allowed opacity-50"
                      : on
                        ? "cursor-pointer border-primary/50 bg-accent/30"
                        : "cursor-pointer border-border hover:bg-muted/40"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={on}
                      disabled={!a.available}
                      onChange={(e) =>
                        setSelected((s) => ({ ...s, [m]: e.target.checked }))
                      }
                      className="size-4 accent-[oklch(0.78_0.14_195)]"
                    />
                    <span className="font-mono text-sm">{m}</span>
                    <span className="ml-auto font-mono text-[10px] text-muted-foreground">
                      {METHOD_META[m].tag}
                    </span>
                  </div>
                  <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
                    {a.reason ?? METHOD_META[m].blurb}
                  </p>
                </label>
              );
            })}
          </div>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="font-mono text-xs text-muted-foreground">
              runs on the same (X, Y) — no permutation Monte Carlo
            </p>
            <Button
              type="button"
              disabled={!yData || busy || selectedAvailable.length === 0}
              onClick={() => void onRun()}
            >
              {busy
                ? "Computing…"
                : `Run selected (${selectedAvailable.length})`}
            </Button>
          </div>
        </CardContent>
      </Card>

      {error && <ErrorBanner title="analysis error" message={error} />}
    </div>
  );
}