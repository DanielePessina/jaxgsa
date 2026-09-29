import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  analyzeCloud,
  analyzeGenerated,
  type AnalysisResult,
  type DesignMethod,
  type GeneratedDesign,
  type XSource,
  type YData,
} from "./engine";
import { ALL_METHODS, METHOD_META, type MethodKey } from "./methods";
import { ErrorBanner } from "./primitives";
import { PanelHeading } from "./PanelHeading";
import { useBusyAction } from "./hooks";
import type { ProblemSpec } from "@/jaxgsa/sampling";

interface MethodAvailability {
  available: boolean;
  reason: string | null;
}

const DESIGN_METHODS: DesignMethod[] = ["sobol", "morris", "kucherenko", "efast"];

const DEFAULT_SELECTION: Record<MethodKey, boolean> = {
  sobol: false,
  morris: false,
  kucherenko: false,
  efast: false,
  pce: true,
  shapley: false,
  borgonovo: false,
  pawn: false,
};

export function availability(
  x: XSource | null,
  problem: ProblemSpec | null,
): Record<MethodKey, MethodAvailability> {
  const dedicated = (method: DesignMethod): MethodAvailability => {
    const available = x?.kind === "design" && x.method === method;

    return {
      available,
      reason: available ? null : `Requires its own ${method} design.`,
    };
  };

  const given = (): MethodAvailability => {
    if (!x) {
      return { available: false, reason: "Load X before choosing this method." };
    }

    const truncated = problem?.marginals.find(
      (m) => m.kind === "gaussian" && (m.low !== undefined || m.high !== undefined),
    );

    if (truncated) {
      return {
        available: false,
        reason: "Browser PCE does not yet support truncated Gaussian inputs.",
      };
    }

    return {
      available: true,
      reason: null,
    };
  };

  return {
    sobol: dedicated("sobol"),
    morris: dedicated("morris"),
    kucherenko: dedicated("kucherenko"),
    efast: dedicated("efast"),
    pce: given(),
    shapley: given(),
    borgonovo: given(),
    pawn: given(),
  };
}

export function AnalyzePanel({
  problem,
  designs,
  yData,
  appendResult,
  xSource,
  setXSource,
  givenX,
  onGoResults,
}: {
  problem: ProblemSpec;
  designs: Partial<Record<DesignMethod, GeneratedDesign>>;
  yData: YData | null;
  appendResult: (r: AnalysisResult) => void;
  xSource: XSource | null;
  setXSource: (x: XSource | null) => void;
  givenX: Float64Array | null;
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

  const avail = availability(xSource, problem);

  // Fall back when the selected design disappears.
  useEffect(() => {
    if (xSource?.kind === "design" && !designs[xSource.method]) {
      const methods = DESIGN_METHODS.filter(
        (method): method is DesignMethod => designs[method] !== undefined,
      );

      const last = methods[methods.length - 1];
      setXSource(last ? { kind: "design", method: last } : null);
    }
  }, [designs, xSource, setXSource]);

  // Selecting a design's X source auto-enables its analysis method.
  useEffect(() => {
    if (xSource?.kind === "design") {
      setSelected(() => ({
        sobol: xSource.method === "sobol",
        morris: xSource.method === "morris",
        kucherenko: xSource.method === "kucherenko",
        efast: xSource.method === "efast",
        pce: false,
        shapley: false,
        borgonovo: false,
        pawn: false,
      }));
    }
  }, [xSource]);

  const selectedAvailable = ALL_METHODS.filter(
    (m) => selected[m] && avail[m].available,
  );

  const compatibleMethods = ALL_METHODS.filter((m) => avail[m].available);

  const onRun = async () => {
    const methods = ALL_METHODS.filter((m) => selected[m] && avail[m].available);

    if (methods.length === 0) {
      setError("Select at least one available method.");

      return;
    }

    const design = xSource?.kind === "design" ? designs[xSource.method] : null;
    const x = design ? design.samples : givenX;

    if (!x || !yData) return;

    setError(null);
    setRunning({ current: 0, total: methods.length, label: methods[0] });

    for (let i = 0; i < methods.length; i++) {
      const m = methods[i];
      setRunning({ current: i, total: methods.length, label: m });
      await new Promise((r) => setTimeout(r, 30)); // let the progress label paint

      try {
        if (m === "pce" || m === "shapley" || m === "borgonovo" || m === "pawn") {
          appendResult(analyzeCloud(m, problem, x, yData, Number(order)));
        } else if (design) {
          appendResult(await analyzeGenerated(design, yData));
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

  const xLabel =
    xSource?.kind === "design"
      ? `X = ${xSource.method} design · ${designs[xSource.method]?.nRuns} runs (from step 02)`
      : xSource?.kind === "uploaded"
        ? givenX
          ? `X = uploaded · ${givenX.length / problem.names.length} rows`
          : "X = uploaded"
        : "no X loaded";

  return (
    <Card>
      <CardContent className="space-y-5 pt-5">
        <PanelHeading
          kicker="analyze"
          title="Which question do you ask of these runs?"
          sub="Each method pairs an estimator with the (X, Y) data chosen in step 02. Unavailable methods explain what they need."
        />

        <p className="font-mono text-xs text-muted-foreground">{xLabel}</p>

        <div>
          {xSource?.kind === "uploaded" && (
            <div className="mb-3 flex flex-wrap items-end gap-4 rounded-sm border border-border/60 bg-muted/20 p-3">
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
              No data selected — go back to step 02 to choose the data source.
            </div>
          ) : (
            <div className="grid gap-3 md:grid-cols-2">
              {ALL_METHODS.map((m) => {
                const on = selected[m] && avail[m].available;
                const enabled = avail[m].available;

                return (
                  <label
                    key={m}
                    className={`rounded-sm border p-3 transition-colors ${
                      !enabled
                        ? "cursor-not-allowed border-border/60 bg-muted/10"
                        : on
                          ? "cursor-pointer border-primary/50 bg-accent/30"
                          : "cursor-pointer border-border hover:bg-muted/40"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={on}
                        disabled={!enabled}
                        onChange={(e) =>
                          setSelected((s) => ({ ...s, [m]: e.target.checked }))
                        }
                        className="size-4 accent-primary"
                      />
                      <span
                        className={`font-mono text-sm ${enabled ? "" : "text-muted-foreground"}`}
                      >
                        {m}
                      </span>
                      <span className="ml-auto flex items-center gap-2">
                        <span
                          className={`rounded-sm border px-1.5 py-0.5 font-mono text-[10px] ${
                            enabled
                              ? "border-primary/40 bg-primary/10 text-primary"
                              : "border-border text-muted-foreground"
                          }`}
                        >
                          {enabled ? "available" : "unavailable"}
                        </span>
                        <span className="font-mono text-[10px] text-muted-foreground">
                          {METHOD_META[m].tag}
                        </span>
                      </span>
                    </div>
                    <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
                      {enabled ? METHOD_META[m].blurb : avail[m].reason}
                    </p>
                  </label>
                );
              })}
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="font-mono text-xs text-muted-foreground">
            {compatibleMethods.length > 1
              ? "selected methods share this (X, Y)"
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

        {error && <ErrorBanner title="analysis error" message={error} />}
      </CardContent>
    </Card>
  );
}
