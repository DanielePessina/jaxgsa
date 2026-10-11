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
import { createRunGuard, parseOrder } from "./analysisGuards";
import { usePersistentState } from "./session";
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

/** Methods that fit a polynomial expansion and so share the order field. */
const ORDER_METHODS: MethodKey[] = ["pce", "shapley"];

/** True when a marginal is a Gaussian cut off at one or both ends. */
function hasTruncatedGaussian(problem: ProblemSpec | null): boolean {
  return (
    problem?.marginals.some(
      (m) => m.kind === "gaussian" && (m.low !== undefined || m.high !== undefined),
    ) ?? false
  );
}

export function availability(
  x: XSource | null,
  problem: ProblemSpec | null,
): Record<MethodKey, MethodAvailability> {
  const dedicated = (method: DesignMethod): MethodAvailability => {
    if (x?.kind === "design" && x.method === method) {
      return { available: true, reason: null };
    }

    if (!x) {
      return { available: false, reason: `Needs a ${method} design. Choose the data in step 02 first.` };
    }

    return {
      available: false,
      reason:
        x.kind === "design"
          ? `Needs a ${method} design. The X you chose comes from a ${x.method} design.`
          : `Needs a ${method} design. Uploaded X cannot be used with it.`,
    };
  };

  const given = (method: MethodKey): MethodAvailability => {
    if (!x) {
      return { available: false, reason: "Needs X and Y. Choose the data in step 02 first." };
    }

    // Only the polynomial methods need an orthogonal basis. PAWN and
    // Borgonovo work on ranks and distributions, so truncation does not
    // matter to them.
    if (ORDER_METHODS.includes(method) && hasTruncatedGaussian(problem)) {
      return {
        available: false,
        reason:
          "Needs Uniform or untruncated Gaussian inputs. The browser does not support truncated Gaussian inputs for this method yet.",
      };
    }

    return { available: true, reason: null };
  };

  return {
    sobol: dedicated("sobol"),
    morris: dedicated("morris"),
    kucherenko: dedicated("kucherenko"),
    efast: dedicated("efast"),
    pce: given("pce"),
    shapley: given("shapley"),
    borgonovo: given("borgonovo"),
    pawn: given("pawn"),
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
  const [order, setOrder] = usePersistentState("analyze-order", "3");

  const [storedSelection, setSelected] = usePersistentState<Record<MethodKey, boolean>>(
    "analyze-methods",
    DEFAULT_SELECTION,
  );

  // A stored selection from an older build may lack newer methods.
  const selected = { ...DEFAULT_SELECTION, ...storedSelection };

  // One token per run: a change to the data cancels the loop below.
  const [runGuard] = useState(createRunGuard);

  const [running, setRunning] = useState<{
    current: number;
    total: number;
    label: string;
  } | null>(null);

  const { busy, error, setError } = useBusyAction();
  const [done, setDone] = useState<string | null>(null);

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

  // Any change to the inputs of a run makes that run stale. Its loop drops
  // the results it has not stored yet. Panels stay mounted, so leaving the
  // stage does not cancel a run: it keeps going and stores its results.
  // The same change clears the old error and "done" message, which describe
  // a run on the old inputs.
  useEffect(() => () => runGuard.cancel(), [runGuard, yData, xSource, designs, givenX, problem]);

  useEffect(() => {
    setError(null);
    setDone(null);
  }, [yData, xSource, designs, givenX, problem, setError]);

  const selectedAvailable = ALL_METHODS.filter(
    (m) => selected[m] && avail[m].available,
  );

  const compatibleMethods = ALL_METHODS.filter((m) => avail[m].available);

  const usesOrder = ORDER_METHODS.some((m) => selected[m] && avail[m].available);
  const orderParse = parseOrder(order);
  const orderError = usesOrder && !orderParse.ok ? orderParse.message : null;
  const locked = running !== null || busy;

  const onRun = async () => {
    const methods = ALL_METHODS.filter((m) => selected[m] && avail[m].available);

    if (methods.length === 0) {
      setError("Select at least one available method.");

      return;
    }

    if (usesOrder && !orderParse.ok) {
      setError(orderParse.message);

      return;
    }

    // The order is only read by the PCE methods, so a bad field cannot reach
    // them: the guard above returned already.
    const orderValue = orderParse.ok ? orderParse.value : 3;
    const design = xSource?.kind === "design" ? designs[xSource.method] : null;
    const x = design ? design.samples : givenX;

    if (!x || !yData) return;

    const token = runGuard.start();
    setError(null);
    setDone(null);
    setRunning({ current: 0, total: methods.length, label: methods[0] });

    for (let i = 0; i < methods.length; i++) {
      const m = methods[i];
      setRunning({ current: i, total: methods.length, label: m });
      await new Promise((r) => setTimeout(r, 30)); // let the progress label paint

      if (!runGuard.isCurrent(token)) {
        setRunning(null);
        setError("The run stopped because the data changed. Run it again.");

        return;
      }

      try {
        if (m === "pce" || m === "shapley" || m === "borgonovo" || m === "pawn") {
          appendResult(analyzeCloud(m, problem, x, yData, orderValue));
        } else if (design) {
          const result = await analyzeGenerated(design, yData);

          // The await can outlast a change to Y: drop the old Y's result.
          if (!runGuard.isCurrent(token)) {
            setRunning(null);
            setError("The run stopped because the data changed. Run it again.");

            return;
          }

          appendResult(result);
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
    setDone(`Finished ${methods.length} method${methods.length === 1 ? "" : "s"}.`);
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
          {usesOrder && (
            <div className="mb-3 flex flex-wrap items-end gap-4 rounded-sm border border-border/60 bg-muted/20 p-3">
              <div className="space-y-1.5">
                <Label htmlFor="order" className="font-mono text-xs">
                  PCE polynomial order
                </Label>
                <Input
                  id="order"
                  type="number"
                  inputMode="numeric"
                  min={1}
                  step={1}
                  value={order}
                  disabled={locked}
                  aria-invalid={orderError !== null}
                  aria-describedby={orderError ? "order-error order-help" : "order-help"}
                  onChange={(e) => setOrder(e.target.value)}
                  className="w-28 font-mono"
                />
              </div>
              <div className="max-w-sm space-y-1 pb-1">
                <p id="order-help" className="text-xs leading-relaxed text-muted-foreground">
                  Shared by PCE and PCE-backed Shapley. Higher orders add basis
                  terms quickly as the parameter count grows. The fit lowers the
                  order when there are too few runs.
                </p>
                {orderError && (
                  <p id="order-error" role="alert" className="text-xs text-destructive">
                    {orderError}
                  </p>
                )}
              </div>
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
                        disabled={!enabled || locked}
                        aria-label={m}
                        aria-describedby={`method-${m}-desc`}
                        onChange={(e) =>
                          setSelected({ ...selected, [m]: e.target.checked })
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
                    <p
                      id={`method-${m}-desc`}
                      className="mt-1.5 text-xs leading-relaxed text-muted-foreground"
                    >
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
            disabled={
              !yData || locked || selectedAvailable.length === 0 || orderError !== null
            }
            onClick={() => void onRun()}
          >
            {running
              ? `Running ${running.label} (${running.current + 1}/${running.total})…`
              : busy
                ? "Computing…"
                : `Run selected (${selectedAvailable.length})`}
          </Button>
        </div>

        {/* Always mounted, so screen readers announce each change. */}
        <p role="status" aria-live="polite" className="sr-only">
          {running
            ? `Running ${running.label}, method ${running.current + 1} of ${running.total}.`
            : (done ?? "")}
        </p>

        {error && <ErrorBanner title="analysis error" message={error} />}
      </CardContent>
    </Card>
  );
}
