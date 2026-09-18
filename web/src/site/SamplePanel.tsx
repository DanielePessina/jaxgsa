import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { downloadText } from "./download";
import {
  buildDesignCsv,
  generateDesign,
  type DesignMethod,
  type GeneratedDesign,
} from "./engine";
import { useBusyAction } from "./hooks";
import { DESIGN_METHOD_META } from "./methods";
import { ErrorBanner, NumberField, ProblemChips } from "./primitives";
import type { ProblemSpec } from "@/jaxgsa/sampling";

const DESIGN_METHODS: DesignMethod[] = ["sobol", "morris"];

function DesignMethodCell({
  method,
  problem,
  design,
  onDesign,
}: {
  method: DesignMethod;
  problem: ProblemSpec;
  design: GeneratedDesign | null;
  onDesign: (gen: GeneratedDesign) => void;
}) {
  const meta = DESIGN_METHOD_META[method];
  const [baseN, setBaseN] = useState("128");
  const [nTrajectories, setNTrajectories] = useState("20");
  const [numLevels, setNumLevels] = useState("4");
  const [seed, setSeed] = useState("0");
  const { busy, error, run } = useBusyAction();

  const onGenerate = () => {
    const config =
      method === "sobol"
        ? { baseN: Number(baseN), calcSecondOrder: false, seed: Number(seed) }
        : {
            nTrajectories: Number(nTrajectories),
            numLevels: Number(numLevels),
            seed: Number(seed),
          };
    void run(() => onDesign(generateDesign(method, problem, config)));
  };

  const D = problem.names.length;
  const plannedRuns =
    method === "sobol"
      ? Number(baseN) * (D + 2)
      : Number(nTrajectories) * (D + 1);
  const preview = design
    ? Array.from({ length: Math.min(3, design.nRuns) }, (_, r) => {
        const row = new Array<number>(D);
        for (let j = 0; j < D; j++) row[j] = design.samples[r * D + j];
        return { id: r, row };
      })
    : [];

  return (
    <div className="flex flex-col gap-3 rounded-sm border border-border p-4">
      <div className="flex items-center justify-between gap-2">
        <h3 className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
          {meta.name}
        </h3>
        <Badge variant="outline" className="font-mono text-[10px]">
          {meta.tag}
        </Badge>
      </div>
      <p className="text-xs leading-relaxed text-muted-foreground">
        {meta.blurb}
      </p>

      <div className="grid grid-cols-2 gap-2">
        {method === "sobol" && (
          <>
            <NumberField
              id={`${method}-baseN`}
              label="Base N"
              value={baseN}
              onChange={setBaseN}
              min={16}
            />
            <div className="flex items-end pb-1">
              <div className="rounded-sm border border-border/70 bg-muted/20 px-2.5 py-1.5 text-xs text-muted-foreground">
                S1 + ST
                <span className="ml-2 font-mono text-[10px]">S2 not ported yet</span>
              </div>
            </div>
          </>
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
        <NumberField
          id={`${method}-seed`}
          label="Seed"
          value={seed}
          onChange={setSeed}
        />
      </div>

      {Number.isFinite(plannedRuns) && plannedRuns > 0 && (
        <div className="flex items-center justify-between rounded-sm border border-border/60 bg-muted/20 px-3 py-2 text-xs">
          <span className="text-muted-foreground">Model evaluation budget</span>
          <span className="font-mono tabular-nums">
            up to {plannedRuns.toLocaleString()} runs
          </span>
        </div>
      )}

      <Button type="button" size="sm" onClick={onGenerate} disabled={busy}>
        {busy ? "Generating…" : "Generate design"}
      </Button>

      {design && (
        <div className="space-y-2 border-t border-border pt-2">
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
        </div>
      )}

      {error && <ErrorBanner title="sampling error" message={error} />}
    </div>
  );
}

export function SamplePanel({
  problem,
  designs,
  onDesign,
  onUseExisting,
}: {
  problem: ProblemSpec | null;
  designs: Partial<Record<DesignMethod, GeneratedDesign>>;
  onDesign: (gen: GeneratedDesign) => void;
  onUseExisting: () => void;
}) {
  const [selectedMethod, setSelectedMethod] = useState<DesignMethod>("sobol");

  if (!problem) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          Define a problem in section 01 first — each design is drawn from
          your problem's marginals.
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      <ProblemChips problem={problem} />
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-normal tracking-tight">
            Choose how you are starting
          </CardTitle>
          <CardDescription>
            Pick the question first. A new experiment uses a method-specific
            design; completed model runs use the given-data route.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
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
                  className={`group rounded-md border p-3 text-left transition-colors ${
                    selected
                      ? "border-primary/55 bg-accent/30"
                      : "border-border bg-background/35 hover:border-border/90 hover:bg-muted/30"
                  }`}
                >
                  <span className="flex items-center justify-between gap-2">
                    <span className="text-sm font-medium capitalize">{method}</span>
                    <span className={`size-2 rounded-full ${selected ? "bg-primary" : "bg-muted-foreground/25"}`} />
                  </span>
                  <span className="mt-2 block text-xs leading-relaxed text-muted-foreground">
                    {method === "sobol"
                      ? "Quantify first-order and total variance. Best general default."
                      : "Screen many inputs with fewer model evaluations."}
                  </span>
                  <span className="mt-3 block font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                    {generated ? "design ready" : method === "sobol" ? "new experiment" : "lower budget"}
                  </span>
                </button>
              );
            })}
            <button
              type="button"
              onClick={onUseExisting}
              className="group rounded-md border border-border bg-background/35 p-3 text-left transition-colors hover:border-border/90 hover:bg-muted/30"
            >
              <span className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium">Existing X/Y</span>
                <span className="text-muted-foreground transition-transform group-hover:translate-x-0.5">→</span>
              </span>
              <span className="mt-2 block text-xs leading-relaxed text-muted-foreground">
                Fit PCE or calculate PCE-backed Shapley from completed runs.
              </span>
              <span className="mt-3 block font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                upload data
              </span>
            </button>
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
              />
            </motion.div>
          </AnimatePresence>
        </CardContent>
      </Card>
    </div>
  );
}
