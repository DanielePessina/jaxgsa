import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { PanelHeading } from "./PanelHeading";
import { FileUpload } from "./FileUpload";
import { downloadText } from "./download";
import {
  alignColumnsByRunId,
  buildDesignCsv,
  generateDesign,
  loadDemoCloud,
  parseXGiven,
  parseYColumns,
  scalarYData,
  describeY,
  type DesignMethod,
  type GeneratedDesign,
  type ParsedCsv,
  type XSource,
  type YData,
} from "./engine";
import { useBusyAction } from "./hooks";
import { DESIGN_METHOD_META } from "./methods";
import { ErrorBanner, NumberField } from "./primitives";
import { SchemaGuide } from "./SchemaGuide";
import type { Demo } from "./demos";
import type { ProblemSpec } from "@/jaxgsa/sampling";

const DESIGN_METHODS: DesignMethod[] = ["sobol", "morris", "kucherenko", "efast"];

/** Which data route the user picked in this panel. */
export type DataRoute = { kind: "new" } | { kind: "existing" };

/**
 * How far the chosen route has progressed. Both routes converge on the same
 * readiness rule — X source chosen and outputs attached — so the App stepper
 * can unlock the analyze step on it.
 */
export interface DataRouteStatus {
  /** The route has both an X source and outputs attached. */
  ready: boolean;
  summary: string;
}

export function dataRouteStatus(
  xSource: XSource | null,
  yData: YData | null,
): DataRouteStatus {
  if (xSource === null) {
    return { ready: false, summary: "no data yet" };
  }

  if (yData === null) {
    return {
      ready: false,
      summary:
        xSource.kind === "design"
          ? "design selected — bring back the model outputs"
          : "X uploaded — outputs missing",
    };
  }

  return { ready: true, summary: `${yData.rowCount} runs ready for analysis` };
}

interface RouteCardProps {
  selected: boolean;
  title: string;
  status?: string;
  blurb: string;
  onSelect: () => void;
}

function RouteCard({ selected, title, status, blurb, onSelect }: RouteCardProps) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={onSelect}
      className={`rounded-md border p-3 text-left transition-colors ${
        selected
          ? "border-primary/55 bg-accent/30"
          : "border-border bg-background/35 hover:border-border/90 hover:bg-muted/30"
      }`}
    >
      <span className="flex items-center justify-between gap-2">
        <span className="inline-flex items-baseline gap-2">
          <span className="text-sm font-medium">{title}</span>
          {status ? (
            <span className="font-mono text-[10px] uppercase tracking-wider text-primary">
              {status}
            </span>
          ) : null}
        </span>
        {selected ? (
          <Check className="size-4 text-primary" />
        ) : (
          <span className="size-2 rounded-full bg-muted-foreground/25" />
        )}
      </span>
      <span className="mt-2 block text-xs leading-relaxed text-muted-foreground">
        {blurb}
      </span>
    </button>
  );
}

/** Config fields + preview + download for one design method. */
function DesignMethodCell({
  method,
  problem,
  design,
  onDesign,
  active,
  onUse,
}: {
  method: DesignMethod;
  problem: ProblemSpec;
  design: GeneratedDesign | null;
  onDesign: (gen: GeneratedDesign) => void;
  /** Whether this method's design is the currently selected X source. */
  active: boolean;
  onUse: () => void;
}) {
  const meta = DESIGN_METHOD_META[method];
  const [baseN, setBaseN] = useState("128");
  const [nTrajectories, setNTrajectories] = useState("20");
  const [numLevels, setNumLevels] = useState("4");
  const [nSamples, setNSamples] = useState("128");
  const [nPerCurve, setNPerCurve] = useState("");
  const [seed, setSeed] = useState("0");
  const { busy, error, run } = useBusyAction();

  const D = problem.names.length;

  // eFAST minimum points per curve: 4*M^2*max(D-1, 1) + 1 with M=4.
  const efastMin = 4 * 16 * Math.max(D - 1, 1) + 1;

  const onGenerate = () => {
    const config =
      method === "sobol"
        ? { baseN: Number(baseN), calcSecondOrder: false, seed: Number(seed) }
        : method === "morris"
          ? {
              nTrajectories: Number(nTrajectories),
              numLevels: Number(numLevels),
              seed: Number(seed),
            }
          : method === "kucherenko"
            ? { nSamples: Number(nSamples), seed: Number(seed) }
            : {
                nPerCurve: Number(nPerCurve || efastMin),
                M: 4,
                seed: Number(seed),
              };

    void run(() => onDesign(generateDesign(method, problem, config)));
  };

  const plannedRuns =
    method === "sobol"
      ? Number(baseN) * (D + 2)
      : method === "morris"
        ? Number(nTrajectories) * (D + 1)
        : method === "kucherenko"
          ? Number(nSamples) * (2 * D + 1)
          : Number(nPerCurve || efastMin) * D;

  const preview = design
    ? Array.from({ length: Math.min(3, design.nRuns) }, (_, r) => {
        const row = Array.from({ length: D }, () => 0);

        for (let j = 0; j < D; j++) row[j] = design.samples[r * D + j];

        return { id: r, row };
      })
    : [];

  return (
    <div className="rounded-sm border border-border p-4">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-medium capitalize">{meta.name}</h3>
        <span className="font-mono text-[10px] text-muted-foreground">
          {meta.tag}
        </span>
      </div>
      <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
        {meta.blurb}
      </p>

      <div className="mt-3 grid grid-cols-2 gap-2">
        {method === "sobol" && (
          <NumberField
            id={`${method}-baseN`}
            label="Base N"
            value={baseN}
            onChange={setBaseN}
            min={16}
          />
        )}
        {method === "morris" && (
          <>
            <NumberField
              id={`${method}-traj`}
              label="Trajectories"
              value={nTrajectories}
              onChange={setNTrajectories}
              min={2}
            />
            <NumberField
              id={`${method}-levels`}
              label="Levels"
              value={numLevels}
              onChange={setNumLevels}
              min={2}
            />
          </>
        )}
        {method === "kucherenko" && (
          <NumberField
            id={`${method}-samples`}
            label="Samples"
            value={nSamples}
            onChange={setNSamples}
            min={2}
          />
        )}
        {method === "efast" && (
          <NumberField
            id={`${method}-curve`}
            label="Points per curve"
            value={nPerCurve || String(efastMin)}
            onChange={setNPerCurve}
            min={efastMin}
          />
        )}
        <NumberField
          id={`${method}-seed`}
          label="Seed"
          value={seed}
          onChange={setSeed}
        />
      </div>

      {Number.isFinite(plannedRuns) && plannedRuns > 0 && (
        <div className="mt-3 flex items-center justify-between rounded-sm border border-border/60 bg-muted/20 px-3 py-2 text-xs">
          <span className="text-muted-foreground">Model evaluation budget</span>
          <span className="font-mono tabular-nums">
            up to {plannedRuns.toLocaleString()} runs
          </span>
        </div>
      )}

      <div className="mt-3 flex items-center gap-2">
        <Button type="button" size="sm" onClick={onGenerate} disabled={busy}>
          {busy ? "Generating…" : design ? "Regenerate design" : "Generate design"}
        </Button>
        {design && !active && (
          <Button type="button" variant="outline" size="sm" onClick={onUse}>
            Use this design
          </Button>
        )}
        {active && (
          <span className="inline-flex items-center gap-1 font-mono text-[11px] text-primary">
            <Check className="size-3" /> in use
          </span>
        )}
      </div>

      {design && (
        <div className="mt-3 space-y-2 border-t border-border pt-3">
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {design.summary.map(([k, v]) => (
              <span key={k} className="font-mono text-xs">
                <span className="text-muted-foreground">{k}: </span>
                <span className="tabular-nums">{v}</span>
              </span>
            ))}
          </div>
          {design.notes.map((n) => (
            <p key={n} className="text-xs text-muted-foreground">
              {n}
            </p>
          ))}
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12 font-mono text-xs">id</TableHead>
                  {problem.names.map((n) => (
                    <TableHead key={n} className="font-mono text-xs">
                      {n}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {preview.map(({ id, row }) => (
                  <TableRow key={id}>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {id}
                    </TableCell>
                    {row.map((v, j) => (
                      <TableCell key={j} className="font-mono text-xs tabular-nums">
                        {v.toFixed(4)}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() =>
              downloadText(
                `jaxgsa-${method}-design.csv`,
                buildDesignCsv(problem, design),
                "text/csv",
              )
            }
            className="w-full"
          >
            Download design CSV
          </Button>
          <p className="text-xs leading-relaxed text-muted-foreground">
            Evaluate your model at these points and upload the outputs below —
            keep the <span className="font-mono">run_id</span> column so the
            outputs can be aligned back onto this design.
          </p>
        </div>
      )}

      {error && <ErrorBanner title="sampling error" message={error} />}
    </div>
  );
}

/**
 * Step 02 — where the (X, Y) data comes from. One panel owns the whole data
 * decision: sample a design for a new experiment, or upload an existing
 * point cloud. The expected file schemas render live beside the uploads so
 * the contract is learned in the same place the files are chosen.
 */
export function DataPanel({
  problem,
  designs,
  route,
  onRouteChange,
  onDesign,
  xSource,
  setXSource,
  yData,
  onYData,
  givenX,
  setGivenX,
  givenXLabel,
  setGivenXLabel,
  activeDemo,
  disabled,
}: {
  problem: ProblemSpec;
  designs: Partial<Record<DesignMethod, GeneratedDesign>>;
  route: DataRoute;
  onRouteChange: (r: DataRoute) => void;
  onDesign: (gen: GeneratedDesign) => void;
  xSource: XSource | null;
  setXSource: (x: XSource | null) => void;
  yData: YData | null;
  /** Central Y setter: also clears results upstream. */
  onYData: (y: YData | null) => void;
  givenX: Float64Array | null;
  setGivenX: (x: Float64Array | null) => void;
  givenXLabel: string;
  setGivenXLabel: (label: string) => void;
  activeDemo: Demo | null;
  disabled: boolean;
}) {
  const [selectedMethod, setSelectedMethod] = useState<DesignMethod>("sobol");
  const [error, setError] = useState<string | null>(null);
  const { busy } = useBusyAction();

  const D = problem.names.length;

  const designMethods = DESIGN_METHODS.filter(
    (m): m is DesignMethod => designs[m] !== undefined,
  );

  const demo = activeDemo;

  const pickDesign = (method: DesignMethod) => {
    setXSource({ kind: "design", method });
    onYData(null);
    setError(null);
  };

  const pickUploaded = (x: Float64Array, label: string) => {
    setXSource({ kind: "uploaded" });
    setGivenX(x);
    setGivenXLabel(label);
    onYData(null);
    setError(null);
  };

  const onUploadX = (parsed: ParsedCsv) => {
    try {
      const flat = parseXGiven(parsed, problem);
      pickUploaded(
        flat,
        `uploaded X · ${flat.length / D} rows × ${D} cols`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const onDemoCloud = () => {
    if (!demo) return;
    const { x, y, n } = loadDemoCloud(demo, 1024);
    pickUploaded(x, `example point cloud (${demo.label}) · ${n} rows`);
    onYData(scalarYData(y, `demo model (${demo.label}) · ${n} outputs`));
  };

  const onUploadY = (parsed: ParsedCsv) => {
    setError(null);

    try {
      const { columns, runIds, rowCount } = parseYColumns(parsed, problem);

      if (xSource?.kind === "design") {
        const design = designs[xSource.method];

        if (!design) {
          throw new Error("the design for the selected X source is missing");
        }

        if (runIds === null) {
          throw new Error(
            'the Y file must contain a "run_id" column matching the design download',
          );
        }

        const aligned = alignColumnsByRunId(columns, runIds, design);
        onYData({ columns: aligned, rowCount, label: `uploaded Y · ${rowCount} runs` });
      } else if (xSource?.kind === "uploaded") {
        const N = givenX ? givenX.length / D : 0;

        if (rowCount !== N) {
          throw new Error(`X has ${N} rows but Y has ${rowCount} values`);
        }

        onYData({ columns, rowCount, label: `uploaded Y · ${rowCount} outputs` });
      } else {
        throw new Error(
          "Attach outputs to a data route first — generate a design or upload X.",
        );
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
    onYData(scalarYData(y, `demo model (${demo.label}) · ${y.length} runs`));
  };

  const { ready, summary } = dataRouteStatus(xSource, yData);

  return (
    <Card>
      <CardContent className="space-y-5 pt-5">
        <PanelHeading
          kicker="data"
          title="Where does your data come from?"
          sub="Two ways in. Start a new experiment: sample a design, evaluate your model at those points, and bring the outputs back. Or continue from model runs you already have."
        />

        <div className="grid gap-2 md:grid-cols-2">
          <RouteCard
            selected={route.kind === "new"}
            title="Start a new experiment"
            status={
              xSource?.kind === "design" ? `${xSource.method} design in use` : undefined
            }
            blurb="Generate a method-specific design, run your model at those points, and bring the outputs back."
            onSelect={() => onRouteChange({ kind: "new" })}
          />
          <RouteCard
            selected={route.kind === "existing"}
            title="I have completed model runs"
            status={xSource?.kind === "uploaded" ? "X loaded" : undefined}
            blurb="Upload inputs X and outputs Y you already have. Fits PCE, Shapley, Borgonovo and PAWN."
            onSelect={() => onRouteChange({ kind: "existing" })}
          />
        </div>

        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={route.kind}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.16, ease: "easeOut" }}
          >
            {route.kind === "new" ? (
              <div className="space-y-3">
                <div className="grid gap-2 md:grid-cols-3">
                  {DESIGN_METHODS.map((method) => {
                    const selected = selectedMethod === method;
                    const generated = designs[method] !== undefined;

                    return (
                      <button
                        key={method}
                        type="button"
                        aria-pressed={selected}
                        onClick={() => setSelectedMethod(method)}
                        className={`rounded-md border p-3 text-left transition-colors ${
                          selected
                            ? "border-primary/55 bg-accent/30"
                            : "border-border bg-background/35 hover:border-border/90 hover:bg-muted/30"
                        }`}
                      >
                        <span className="flex items-center justify-between gap-2">
                          <span className="text-sm font-medium capitalize">
                            {method}
                          </span>
                          <span
                            className={`size-2 rounded-full ${selected ? "bg-primary" : "bg-muted-foreground/25"}`}
                          />
                        </span>
                        <span className="mt-2 block text-xs leading-relaxed text-muted-foreground">
                          {method === "sobol"
                            ? "Quantify first-order and total variance. Best general default."
                            : method === "morris"
                              ? "Screen many inputs with fewer model evaluations."
                              : method === "kucherenko"
                                ? "Variance decomposition on a conditional-copula design (independent inputs only)."
                                : "Deterministic Fourier-spectrum S1/ST from sinusoidal search curves."}
                        </span>
                        <span className="mt-3 block font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                          {generated
                            ? "design ready"
                            : method === "sobol"
                              ? "new experiment"
                              : method === "morris"
                                ? "lower budget"
                                : method === "kucherenko"
                                  ? "cross-check"
                                  : "deterministic"}
                        </span>
                      </button>
                    );
                  })}
                </div>

                <AnimatePresence mode="wait" initial={false}>
                  <motion.div
                    key={selectedMethod}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.16, ease: "easeOut" }}
                  >
                    <DesignMethodCell
                      method={selectedMethod}
                      problem={problem}
                      design={designs[selectedMethod] ?? null}
                      onDesign={onDesign}
                      active={
                        xSource?.kind === "design" && xSource.method === selectedMethod
                      }
                      onUse={() => pickDesign(selectedMethod)}
                    />
                  </motion.div>
                </AnimatePresence>

                {xSource?.kind === "design" && (
                  <div className="space-y-2 border-t border-border pt-3">
                    <FileUpload
                      label="Upload outputs (Y)"
                      hint="must include run_id — the row numbers 0…N−1 from the downloaded design CSV · outputs: y, or name_t0, name_t0.5… for time-resolved data"
                      disabled={disabled}
                      onLoaded={onUploadY}
                      onError={setError}
                      exampleLabel={
                        demo ? `Evaluate demo model (${demo.label})` : undefined
                      }
                      onExample={demo ? onDemoY : undefined}
                      exampleDisabled={disabled}
                    />
                    <p className="font-mono text-xs text-muted-foreground">
                      {yData
                        ? `${yData.label} · ${describeY(yData)}`
                        : `X = ${xSource.method} design · ${designs[xSource.method]?.nRuns} runs`}
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4 rounded-sm border border-border p-4">
                <FileUpload
                  label="Upload inputs (X)"
                  hint={`header must be exactly: ${problem.names.join(", ")} — one row per run, in order. No run_id.`}
                  disabled={disabled || busy}
                  onLoaded={onUploadX}
                  onError={setError}
                  exampleLabel={
                    demo ? `Try PCE/Shapley example (${demo.label})` : undefined
                  }
                  onExample={demo ? onDemoCloud : undefined}
                  exampleDisabled={disabled || busy}
                />
                <FileUpload
                  label="Upload outputs (Y)"
                  hint="no run_id needed — one output column per output slice (y, or name_t0, name_t0.5…), one value per X row, same row order"
                  disabled={disabled || givenX === null}
                  onLoaded={onUploadY}
                  onError={setError}
                />
                <p className="font-mono text-xs text-muted-foreground">
                  {yData
                    ? `${yData.label} · ${describeY(yData)}`
                    : givenXLabel}
                </p>
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        <SchemaGuide
          problem={problem}
          route={
            xSource?.kind === "design"
              ? "design"
              : xSource?.kind === "uploaded"
                ? "uploaded"
                : route.kind === "new"
                  ? "design"
                  : "none"
          }
          method={xSource?.kind === "design" ? xSource.method : undefined}
        />

        {designMethods.length > 0 && (
          <p className="font-mono text-[11px] text-muted-foreground">
            designs in this session: {designMethods.join(", ")}
          </p>
        )}

        <p className="text-xs text-muted-foreground">
          {ready
            ? "Data is ready — continue to the analysis."
            : `Not ready yet: ${summary}.`}
        </p>

        {error && <ErrorBanner title="data error" message={error} />}
      </CardContent>
    </Card>
  );
}
