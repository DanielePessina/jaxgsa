import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
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
import { PanelHeading } from "./PanelHeading";
import { FileUpload } from "./FileUpload";
import { downloadText } from "./download";
import {
  alignColumnsByRunId,
  alignColumnsToXRunIds,
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
import {
  DEFAULT_DESIGN_FORM,
  designConfig,
  designFormErrors,
  designMismatch,
  duplicatesNote,
  efastDefaultPoints,
  efastMinPoints,
  mergeStoredForm,
  type StoredForm,
  plannedRuns,
  type DesignForm,
} from "./designForm";
import { useBusyAction } from "./hooks";
import { DESIGN_METHOD_META } from "./methods";
import { ErrorBanner } from "./primitives";
import { SchemaGuide } from "./SchemaGuide";
import { usePersistentState } from "./session";
import type { Demo } from "./demos";
import type { ProblemSpec } from "@/jaxgsa/sampling";

/**
 * The run_id of each row of an uploaded X, keyed by the X array itself. The
 * ids follow the array, so they survive this panel unmounting while the
 * parent still holds the X.
 */
const X_RUN_IDS = new WeakMap<Float64Array, number[] | null>();

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

/** Labeled number input with an inline hint and error. */
function FormField({
  id,
  label,
  value,
  onChange,
  hint,
  error,
  placeholder,
  min,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  hint?: string;
  error?: string;
  placeholder?: string;
  min?: number;
}) {
  const noteId = `${id}-note`;
  const note = error ?? hint;

  return (
    <div className="space-y-1">
      <Label htmlFor={id} className="font-mono text-xs">
        {label}
      </Label>
      <Input
        id={id}
        type="number"
        min={min}
        step={1}
        value={value}
        placeholder={placeholder}
        aria-invalid={error ? true : undefined}
        aria-describedby={note ? noteId : undefined}
        onChange={(e) => onChange(e.target.value)}
        className="h-8 font-mono text-xs"
      />
      {note && (
        <p
          id={noteId}
          className={`text-[11px] leading-snug ${error ? "text-destructive" : "text-muted-foreground"}`}
        >
          {note}
        </p>
      )}
    </div>
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
  form,
  setField,
  disabled,
}: {
  method: DesignMethod;
  problem: ProblemSpec;
  design: GeneratedDesign | null;
  onDesign: (gen: GeneratedDesign) => void;
  /** Whether this method's design is the currently selected X source. */
  active: boolean;
  onUse: () => void;
  form: DesignForm;
  setField: (field: keyof DesignForm, value: string) => void;
  disabled: boolean;
}) {
  const meta = DESIGN_METHOD_META[method];
  const { busy, error, setError, run } = useBusyAction();

  // The panel stays mounted when the problem changes: drop the old error.
  const [seenProblem, setSeenProblem] = useState(problem);

  if (seenProblem !== problem) {
    setSeenProblem(problem);
    setError(null);
  }

  const D = problem.names.length;
  const errors = designFormErrors(method, form, D);
  const invalid = Object.keys(errors).length > 0;
  const efastMin = efastMinPoints(D);
  const efastDefault = efastDefaultPoints(D);

  const onGenerate = () => {
    if (invalid) return;
    void run(() => onDesign(generateDesign(method, problem, designConfig(method, form, D))));
  };

  const planned = plannedRuns(method, form, D);
  const mismatch = design ? designMismatch(method, form, design, D, active) : null;
  const dupNote = design ? duplicatesNote(design) : null;

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
          <FormField
            id={`${method}-baseN`}
            label="Base N"
            value={form.baseN}
            onChange={(v) => setField("baseN", v)}
            hint="Must be a power of 2: 16, 32, 64, 128…"
            error={errors.baseN}
            min={16}
          />
        )}
        {method === "morris" && (
          <>
            <FormField
              id={`${method}-traj`}
              label="Trajectories"
              value={form.nTrajectories}
              onChange={(v) => setField("nTrajectories", v)}
              error={errors.nTrajectories}
              min={2}
            />
            <FormField
              id={`${method}-levels`}
              label="Levels"
              value={form.numLevels}
              onChange={(v) => setField("numLevels", v)}
              error={errors.numLevels}
              min={2}
            />
          </>
        )}
        {method === "kucherenko" && (
          <FormField
            id={`${method}-samples`}
            label="Model runs"
            value={form.nSamples}
            onChange={(v) => setField("nSamples", v)}
            hint="Minimum number of model runs. Base N is the smallest power of 2 that reaches it, so the design may have more runs."
            error={errors.nSamples}
            min={2}
          />
        )}
        {method === "efast" && (
          <FormField
            id={`${method}-curve`}
            label="Points per curve"
            value={form.nPerCurve}
            onChange={(v) => setField("nPerCurve", v)}
            placeholder={String(efastDefault)}
            hint={`Leave empty for ${efastDefault}. The minimum is ${efastMin}. Fewer points can bias the indices.`}
            error={errors.nPerCurve}
            min={efastMin}
          />
        )}
        <FormField
          id={`${method}-seed`}
          label="Seed"
          value={form.seed}
          onChange={(v) => setField("seed", v)}
          error={errors.seed}
        />
      </div>

      {planned !== null && planned > 0 && (
        <div className="mt-3 rounded-sm border border-border/60 bg-muted/20 px-3 py-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Model evaluation budget</span>
            <span className="font-mono tabular-nums">
              up to {planned.toLocaleString()} runs
            </span>
          </div>
          {(method === "morris" || method === "sobol") && (
            <p className="mt-1 text-[11px] text-muted-foreground">
              The design removes duplicate points, so the real count can be lower.
            </p>
          )}
        </div>
      )}

      {mismatch && (
        <p role="status" className="mt-2 text-xs text-amber-600 dark:text-amber-400">
          {mismatch}
        </p>
      )}

      <div className="mt-3 flex items-center gap-2">
        <Button type="button" size="sm" onClick={onGenerate} disabled={busy || disabled || invalid}>
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
          {dupNote && <p className="text-xs text-muted-foreground">{dupNote}</p>}
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
            onClick={() => {
              try {
                downloadText(
                  `jaxgsa-${method}-design.csv`,
                  buildDesignCsv(problem, design),
                  "text/csv",
                );
              } catch (err) {
                setError(err instanceof Error ? err.message : String(err));
              }
            }}
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
  const [storedMethod, setSelectedMethod] = usePersistentState<DesignMethod>(
    "data-method",
    "sobol",
  );

  const [storedForm, setStoredForm] = usePersistentState<StoredForm>(
    "data-design-form",
    DEFAULT_DESIGN_FORM,
  );

  const [error, setError] = useState<string | null>(null);

  // The panel stays mounted when the problem changes: drop the old error.
  const [seenProblem, setSeenProblem] = useState(problem);

  if (seenProblem !== problem) {
    setSeenProblem(problem);
    setError(null);
  }

  // A stale or edited entry must not break the form: keep only string fields.
  const selectedMethod = DESIGN_METHODS.includes(storedMethod) ? storedMethod : "sobol";
  const form = mergeStoredForm(storedForm);

  const setField = (field: keyof DesignForm, value: string) =>
    setStoredForm((f) => ({ ...mergeStoredForm(f), [field]: value }));

  const D = problem.names.length;

  const designMethods = DESIGN_METHODS.filter(
    (m): m is DesignMethod => designs[m] !== undefined,
  );

  const demo = activeDemo;

  // A design and an uploaded X never coexist: the design route needs run_id
  // in Y, the uploaded route does not. Switching clears the other side.
  const dropUploadedX = () => {
    setGivenX(null);
    setGivenXLabel("no X data loaded");
  };

  const pickDesign = (method: DesignMethod) => {
    setXSource({ kind: "design", method });
    dropUploadedX();
    onYData(null);
    setError(null);
  };

  const handleDesign = (gen: GeneratedDesign) => {
    onDesign(gen);
    dropUploadedX();
    setError(null);
  };

  const pickUploaded = (x: Float64Array, label: string) => {
    setXSource({ kind: "uploaded" });
    setGivenX(x);
    setGivenXLabel(label);
    onYData(null);
    setError(null);
  };

  // Both upload handlers throw on invalid files. FileUpload turns the throw
  // into an error and keeps its success state hidden.
  const onUploadX = (parsed: ParsedCsv, file: File) => {
    const { flat, runIds } = parseXGiven(parsed, problem);

    X_RUN_IDS.set(flat, runIds);

    pickUploaded(
      flat,
      `uploaded X · ${file.name} · ${flat.length / D} rows × ${D} cols${runIds ? " · with run_id" : ""}`,
    );
  };

  // A failed Y upload clears the Y that was loaded before, so the panel never
  // shows an old file next to an error about a new one.
  const onYError = (message: string) => {
    setError(message);
    onYData(null);
  };

  const onDemoCloud = () => {
    if (!demo) return;
    const { x, y, n } = loadDemoCloud(demo, 1024);

    X_RUN_IDS.set(x, null);
    pickUploaded(x, `example point cloud (${demo.label}) · ${n} rows`);
    onYData(scalarYData(y, `demo model (${demo.label}) · ${n} rows`));
  };

  const onUploadY = (parsed: ParsedCsv, file: File) => {
    setError(null);

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

      onYData({
        columns: aligned,
        rowCount,
        label: `uploaded Y · ${file.name} · ${rowCount} rows`,
      });
    } else if (xSource?.kind === "uploaded" && givenX !== null) {
      const N = givenX.length / D;
      const xIds = X_RUN_IDS.get(givenX) ?? null;
      const base = `uploaded Y · ${file.name} · ${rowCount} rows`;

      if (xIds !== null && runIds !== null) {
        // Both files carry run_id: match Y to X by id, in any row order.
        onYData({
          columns: alignColumnsToXRunIds(columns, runIds, xIds),
          rowCount: N,
          label: `${base} · matched to X by run_id`,
        });

        return;
      }

      if (rowCount !== N) {
        throw new Error(`X has ${N} rows but Y has ${rowCount} values`);
      }

      onYData({
        columns,
        rowCount,
        label:
          xIds !== null
            ? `${base} · paired with X by row order (Y has no run_id)`
            : base,
      });
    } else {
      throw new Error(
        "Attach outputs to a data route first — generate a design or upload X.",
      );
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
  const uploadedXActive = xSource?.kind === "uploaded" && givenX !== null;
  const xRunIds = givenX !== null && uploadedXActive ? (X_RUN_IDS.get(givenX) ?? null) : null;

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
                      onDesign={handleDesign}
                      form={form}
                      setField={setField}
                      disabled={disabled}
                      active={
                        xSource?.kind === "design" && xSource.method === selectedMethod
                      }
                      onUse={() => pickDesign(selectedMethod)}
                    />
                  </motion.div>
                </AnimatePresence>

                {xSource?.kind === "uploaded" && (
                  <p className="text-xs text-muted-foreground">
                    Your own X is in use (the other route). Press "Use this design" or
                    generate a design to switch. That removes the uploaded X.
                  </p>
                )}

                {xSource?.kind === "design" && (
                  <div className="space-y-2 border-t border-border pt-3">
                    <FileUpload
                      label="Upload outputs (Y)"
                      hint="must include run_id — the row numbers 0…N−1 from the downloaded design CSV · outputs: y, or name_t0, name_t0.5… for time-resolved data"
                      disabled={disabled}
                      onLoaded={onUploadY}
                      onError={onYError}
                      status={yData ? yData.label : null}
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
                {xSource?.kind === "design" && (
                  <p className="text-xs text-muted-foreground">
                    A {xSource.method} design is in use. Uploading X replaces it.
                  </p>
                )}
                <FileUpload
                  label="Upload inputs (X)"
                  hint={`header must be exactly: ${problem.names.join(", ")} — one row per run. A run_id column from a design download is allowed. If X has one, Y is matched to it by id.`}
                  disabled={disabled}
                  onLoaded={onUploadX}
                  onError={setError}
                  status={xSource?.kind === "uploaded" && givenX ? givenXLabel : null}
                  exampleLabel={
                    demo ? `Try PCE/Shapley example (${demo.label})` : undefined
                  }
                  onExample={demo ? onDemoCloud : undefined}
                  exampleDisabled={disabled}
                />
                <FileUpload
                  label="Upload outputs (Y)"
                  hint={
                    !uploadedXActive
                      ? "upload X first — this slot opens when your own X is loaded"
                      : xRunIds !== null
                        ? "one output column per output slice (y, or name_t0, name_t0.5…), one value per X row. Include run_id to match X by id, in any order. Without run_id, Y is paired with X by row order."
                        : "one output column per output slice (y, or name_t0, name_t0.5…), one value per X row, same row order as X. X has no run_id, so a run_id column in Y is ignored."
                  }
                  disabled={disabled || !uploadedXActive}
                  onLoaded={onUploadY}
                  onError={onYError}
                  status={uploadedXActive && yData ? yData.label : null}
                />
                <p className="font-mono text-xs text-muted-foreground">
                  {!uploadedXActive
                    ? givenXLabel
                    : yData
                      ? `${yData.label} · ${describeY(yData)}`
                      : `Active X: ${givenXLabel}`}
                </p>
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        <SchemaGuide
          problem={problem}
          route={route.kind === "new" ? "design" : "uploaded"}
          method={
            route.kind === "new"
              ? xSource?.kind === "design"
                ? xSource.method
                : selectedMethod
              : undefined
          }
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
