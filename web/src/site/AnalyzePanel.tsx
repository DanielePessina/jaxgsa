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
import { FileUpload } from "./FileUpload";
import { demoForProblem, type Demo } from "./demos";
import {
  alignColumnsByRunId,
  analyzeCloud,
  analyzeGenerated,
  describeY,
  loadDemoCloud,
  parseXGiven,
  parseYColumns,
  scalarYData,
  type AnalysisResult,
  type DesignMethod,
  type GeneratedDesign,
  type ParsedCsv,
  type XSource,
  type YData,
} from "./engine";
import { ALL_METHODS, METHOD_META, type MethodKey } from "./methods";
import { ErrorBanner, MonoSelect } from "./primitives";
import { useBusyAction } from "./hooks";
import type { ProblemSpec } from "@/jaxgsa/sampling";

interface MethodAvailability {
  available: boolean;
  reason: string | null;
}

const DEFAULT_SELECTION: Record<MethodKey, boolean> = {
  sobol: false,
  morris: false,
  pce: true,
  shapley: false,
};

function availability(
  x: XSource | null,
): Record<MethodKey, MethodAvailability> {
  return Object.fromEntries(
    ALL_METHODS.map((method) => {
      const input = METHOD_META[method].input;

      if (input.kind === "dedicated") {
        const available = x?.kind === "design" && x.method === input.design;

        return [method, {
          available,
          reason: available
            ? null
            : `Requires its own ${input.design} design.`,
        }];
      }

      const available = x?.kind === "uploaded";

      return [method, {
        available,
        reason: available
          ? null
          : "Uses an ordinary uploaded X/Y point cloud, not a dedicated estimator design.",
      }];
    }),
  ) as Record<MethodKey, MethodAvailability>;
}

export function AnalyzePanel({
  problem,
  designs,
  yData,
  setYData,
  appendResult,
  xSource,
  setXSource,
  givenX,
  setGivenX,
  givenXLabel,
  setGivenXLabel,
  activeDemo,
  onGoSample,
  onGoResults,
}: {
  problem: ProblemSpec | null;
  designs: Partial<Record<DesignMethod, GeneratedDesign>>;
  yData: YData | null;
  setYData: (y: YData | null) => void;
  appendResult: (r: AnalysisResult) => void;
  xSource: XSource | null;
  setXSource: (x: XSource | null) => void;
  givenX: Float64Array | null;
  setGivenX: (x: Float64Array | null) => void;
  givenXLabel: string;
  setGivenXLabel: (label: string) => void;
  activeDemo: Demo | null;
  onGoSample: () => void;
  onGoResults: () => void;
}) {
  const [order, setOrder] = useState("3");

  const [selected, setSelected] = useState<Record<MethodKey, boolean>>({
    ...DEFAULT_SELECTION,
  });

  const [running, setRunning] = useState<{
    current: number;
    total: number;
    label: string;
  } | null>(null);

  const { busy, error, setError } = useBusyAction();

  const demo = activeDemo ?? demoForProblem(problem);
  const avail = availability(xSource);
  const designMethods = Object.keys(designs) as DesignMethod[];

  // Fall back when the selected design disappears.
  useEffect(() => {
    if (xSource?.kind === "design" && !designs[xSource.method]) {
      const methods = Object.keys(designs) as DesignMethod[];
      const last = methods[methods.length - 1];
      setXSource(last ? { kind: "design", method: last } : null);
      setYData(null);
    }
  }, [designs, xSource, setXSource, setYData]);

  // Selecting a design's X source auto-enables its analysis method.
  useEffect(() => {
    if (xSource?.kind === "design") {
      setSelected(() => ({
        sobol: xSource.method === "sobol",
        morris: xSource.method === "morris",
        pce: false,
        shapley: false,
      }));
    }
  }, [xSource]);

  const pickDesign = (method: DesignMethod) => {
    setXSource({ kind: "design", method });
    setGivenX(null);
    setYData(null);
    setError(null);
  };

  const pickUploaded = (x: Float64Array, label: string) => {
    setXSource({ kind: "uploaded" });
    setGivenX(x);
    setGivenXLabel(label);
    setSelected({ sobol: false, morris: false, pce: true, shapley: false });
    setYData(null);
    setError(null);
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
    pickUploaded(x, `example point cloud (${demo.label}) · ${n} rows`);
    setYData(scalarYData(y, `demo model (${demo.label}) · ${n} outputs`));
  };

  const onUploadY = (parsed: ParsedCsv) => {
    if (!problem) return;
    setError(null);

    try {
      const { columns, runIds, rowCount } = parseYColumns(parsed, problem);

      if (xSource?.kind === "design") {
        const design = designs[xSource.method];

        if (!design) return;

        if (runIds === null) {
          throw new Error(
            'the Y file must contain a "run_id" column matching the design download',
          );
        }

        const aligned = alignColumnsByRunId(columns, runIds, design);
        setYData({ columns: aligned, rowCount, label: `uploaded Y · ${rowCount} runs` });
      } else {
        const N = givenX ? givenX.length / problem.names.length : 0;

        if (rowCount !== N) {
          throw new Error(`X has ${N} rows but Y has ${rowCount} values`);
        }

        setYData({ columns, rowCount, label: `uploaded Y · ${rowCount} outputs` });
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
    setYData(scalarYData(y, `demo model (${demo.label}) · ${y.length} runs`));
  };

  const selectedAvailable = ALL_METHODS.filter(
    (m) => selected[m] && avail[m].available,
  );

  const compatibleMethods = ALL_METHODS.filter((m) => avail[m].available);

  const onRun = async () => {
    if (!problem || !yData) return;
    const methods = ALL_METHODS.filter((m) => selected[m] && avail[m].available);

    if (methods.length === 0) {
      setError("Select at least one available method.");

      return;
    }

    const design = xSource?.kind === "design" ? designs[xSource.method] : null;
    const x = design ? design.samples : givenX;

    if (!x) return;

    setError(null);
    setRunning({ current: 0, total: methods.length, label: methods[0] });

    for (let i = 0; i < methods.length; i++) {
      const m = methods[i];
      setRunning({ current: i, total: methods.length, label: m });
      await new Promise((r) => setTimeout(r, 30)); // let the progress label paint

      try {
        if (m === "pce" || m === "shapley") {
          appendResult(analyzeCloud(m, problem, x, yData, Number(order)));
        } else if (design) {
          appendResult(analyzeGenerated(design, yData));
        }
      } catch (err) {
        setError(
          `${m}: ${err instanceof Error ? err.message : String(err)}`,
        );
        setRunning(null);

        return;
      }
    }

    setRunning(null);
    onGoResults();
  };

  const onResetMethods = () => setSelected({ ...DEFAULT_SELECTION });

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

  const activeDesign =
    xSource?.kind === "design" ? designs[xSource.method] ?? null : null;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base font-normal tracking-tight">
            Input data
          </CardTitle>
          <CardDescription>
            Choose a dedicated design for its paired estimator, or upload an
            ordinary point cloud for the given-data methods. The workbench
            keeps those routes separate so a shape-compatible array is not
            mistaken for a method-compatible design.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-[auto_minmax(0,1fr)] md:items-start">
            <div className="space-y-2">
              <span className="block text-xs font-medium text-muted-foreground">X source</span>
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
                    pickUploaded(givenX, givenXLabel);
                  } else {
                    setError(
                      "Upload an X CSV/Parquet (or load a demo cloud) to use your own data.",
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
              <p className="max-w-52 font-mono text-[11px] leading-relaxed text-muted-foreground">
                {xLabel}
              </p>
            </div>
            <FileUpload
              label="Upload X"
              hint={`header: ${problem.names.join(", ")} — N rows`}
              disabled={busy || running !== null}
              onLoaded={onUploadX}
              onError={setError}
              exampleLabel={demo ? `Try PCE/Shapley example (${demo.label})` : undefined}
              onExample={demo ? onDemoCloud : undefined}
              exampleDisabled={busy || running !== null}
            />
          </div>

          {activeDesign && activeDesign.summary.length > 0 && (
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-sm border border-border/60 bg-muted/20 px-3 py-2">
              <span className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
                {activeDesign.method} settings
              </span>
              {activeDesign.summary.map(([k, v]) => (
                <span key={k} className="font-mono text-xs">
                  <span className="text-muted-foreground">{k}: </span>
                  <span className="tabular-nums">{v}</span>
                </span>
              ))}
            </div>
          )}

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
            <FileUpload
              label="Upload outputs (Y)"
              hint={
                xSource?.kind === "design"
                  ? "CSV/Parquet with run_id (any order) + outputs — y, or name_t0, name_t0.5… for time-resolved data"
                  : "one column per output slice — y, or name_t0, name_t0.5… — one value per X row"
              }
              disabled={!xSource || busy || running !== null}
              onLoaded={onUploadY}
              onError={setError}
              exampleLabel={
                demo && xSource?.kind === "design"
                  ? `Evaluate demo model (${demo.label})`
                  : undefined
              }
              onExample={demo && xSource?.kind === "design" ? onDemoY : undefined}
              exampleDisabled={busy || running !== null}
            />
            <p className="mt-2 font-mono text-xs text-muted-foreground">
              {yData ? `${yData.label} · ${describeY(yData)}` : "no outputs loaded"}
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <div>
            <CardTitle className="text-base font-normal tracking-tight">
              Methods
            </CardTitle>
            <CardDescription>
              Only analyses compatible with the active X source are shown.
              Given-data methods may be combined; dedicated designs stay
              paired with their estimator.
            </CardDescription>
          </div>
          {compatibleMethods.length > 1 && (
            <Button type="button" variant="ghost" size="sm" onClick={onResetMethods}>
              reset
            </Button>
          )}
        </CardHeader>
        <CardContent className="space-y-4">
          {xSource?.kind === "uploaded" && (
            <div className="flex flex-wrap items-end gap-4 rounded-sm border border-border/60 bg-muted/20 p-3">
              <div className="space-y-1.5">
                <Label htmlFor="order" className="font-mono text-xs">
                  PCE polynomial order
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
              <p className="max-w-sm pb-1 text-xs leading-relaxed text-muted-foreground">
                Shared by PCE and PCE-backed Shapley. Higher orders add basis
                terms quickly as the parameter count grows.
              </p>
            </div>
          )}
          {xSource === null ? (
            <div className="rounded-sm border border-dashed border-border px-4 py-6 text-center text-sm text-muted-foreground">
              Select a sampled design or upload X to see compatible analyses.
            </div>
          ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {compatibleMethods.map((m) => {
              const on = selected[m];

              return (
                <label
                  key={m}
                  className={`rounded-sm border p-3 transition-colors ${
                    on
                        ? "cursor-pointer border-primary/50 bg-accent/30"
                        : "cursor-pointer border-border hover:bg-muted/40"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={on}
                      onChange={(e) =>
                        setSelected((s) => ({ ...s, [m]: e.target.checked }))
                      }
                      className="size-4 accent-[oklch(0.78_0.14_195)]"
                    />
                    <span className="font-mono text-sm">{m}</span>
                    <span className="ml-auto flex items-center gap-2">
                      <span
                        className={`rounded-sm border px-1.5 py-0.5 font-mono text-[10px] ${
                          "border-primary/40 bg-primary/10 text-primary"
                        }`}
                      >
                        compatible
                      </span>
                      <span className="font-mono text-[10px] text-muted-foreground">
                        {METHOD_META[m].tag}
                      </span>
                    </span>
                  </div>
                  <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
                    {METHOD_META[m].blurb}
                  </p>
                </label>
              );
            })}
          </div>
          )}
          {xSource && (
            <p className="text-xs leading-relaxed text-muted-foreground">
              {xSource.kind === "design"
                ? `This ${xSource.method} design is paired with ${xSource.method} analysis. To use PCE or Shapley, switch to “my own X” and upload an ordinary point cloud.`
                : "PCE and PCE-backed Shapley can share this uploaded point cloud. Dedicated Sobol and Morris estimators require designs generated in section 02."}
            </p>
          )}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="font-mono text-xs text-muted-foreground">
              {compatibleMethods.length > 1
                ? "selected methods share this uploaded (X, Y)"
                : "design and estimator are paired"}
            </p>
            <Button
              type="button"
              disabled={!yData || busy || running !== null || selectedAvailable.length === 0}
              onClick={() => void onRun()}
            >
              {running
                ? `Running ${running.label} (${running.current + 1}/${running.total})…`
                : busy
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
