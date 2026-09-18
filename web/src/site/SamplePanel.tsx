import { useState } from "react";
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
  const [secondOrder, setSecondOrder] = useState(false);
  const [nTrajectories, setNTrajectories] = useState("20");
  const [numLevels, setNumLevels] = useState("4");
  const [seed, setSeed] = useState("0");
  const { busy, error, run } = useBusyAction();

  const onGenerate = () => {
    const config =
      method === "sobol"
        ? { baseN: Number(baseN), calcSecondOrder: secondOrder, seed: Number(seed) }
        : {
            nTrajectories: Number(nTrajectories),
            numLevels: Number(numLevels),
            seed: Number(seed),
          };
    void run(() => onDesign(generateDesign(method, problem, config)));
  };

  const D = problem.names.length;
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
              <label className="flex cursor-pointer items-center gap-1.5 text-xs">
                <input
                  type="checkbox"
                  checked={secondOrder}
                  onChange={(e) => setSecondOrder(e.target.checked)}
                  className="size-3.5 accent-[oklch(0.78_0.14_195)]"
                />
                Second order
              </label>
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
}: {
  problem: ProblemSpec | null;
  designs: Partial<Record<DesignMethod, GeneratedDesign>>;
  onDesign: (gen: GeneratedDesign) => void;
}) {
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
          <CardTitle className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
            design methods
          </CardTitle>
          <CardDescription>
            Each method draws its own design from your problem's marginals.
            Sampling and analysis are linked: a method can only analyze the
            design it generated. PCE and Shapley instead use an ordinary
            uploaded X/Y point cloud.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {DESIGN_METHODS.map((m) => (
              <DesignMethodCell
                key={m}
                method={m}
                problem={problem}
                design={designs[m] ?? null}
                onDesign={onDesign}
              />
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
