import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { type Demo } from "./demos";
import { downloadText } from "./download";
import {
  buildSessionJson,
  resultToCsv,
  yColumnLabel,
  type AnalysisResult,
  type DesignMethod,
  type GeneratedDesign,
  type YData,
} from "./engine";
import { IndicesChart } from "./IndicesChart";
import { ResultsTable } from "./ResultsTable";
import type { ProblemSpec } from "@/jaxgsa/sampling";

function referenceLine(
  result: AnalysisResult,
  demo: Demo | null,
): string | null {
  if (result.method === "morris") return null;

  if (!demo) return null;
  const s1 = demo.reference.S1.map((v) => v.toFixed(3)).join(", ");
  const st = demo.reference.ST.map((v) => v.toFixed(3)).join(", ");

  return `reference (${demo.label}, analytical): S1 ≈ ${s1} · ST ≈ ${st}`;
}

function SliceTabs({
  slices,
  active,
  onSelect,
}: {
  slices: AnalysisResult["slices"];
  active: number;
  onSelect: (i: number) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1 rounded-full border border-border bg-muted/30 p-1">
      {slices.map((s, i) => (
        <button
          key={`${s.output}-${s.time}`}
          type="button"
          onClick={() => onSelect(i)}
          className={`relative rounded-full px-3 py-1 font-mono text-xs transition-colors ${
            i === active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {i === active && (
            <motion.span
              layoutId={`slice-pill-${s.output}-${s.time}`}
              className="absolute inset-0 rounded-full bg-primary"
              transition={{ type: "spring", stiffness: 400, damping: 32 }}
            />
          )}
          <span className="relative">{yColumnLabel(s.output, s.time)}</span>
        </button>
      ))}
    </div>
  );
}

function ResultCard({
  result,
  demo,
  index,
}: {
  result: AnalysisResult;
  demo: Demo | null;
  index: number;
}) {
  const [sliceIndex, setSliceIndex] = useState(0);
  const ref = referenceLine(result, demo);
  const settings = Object.entries(result.settings);
  const multi = result.slices.length > 1;
  const fileName = `jaxgsa-${result.method}-${index}`;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ type: "spring", stiffness: 260, damping: 30 }}
    >
      <Card>
        <CardContent className="space-y-4 pt-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
              <h3 className="text-base font-medium tracking-tight">
                {result.method}
              </h3>
              {settings.length > 0 && (
                <span className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-muted-foreground">
                  {settings.map(([k, v]) => (
                    <span key={k}>
                      {k}: <span className="tabular-nums">{String(v)}</span>
                    </span>
                  ))}
                </span>
              )}
            </div>
            {multi && (
              <span className="font-mono text-[11px] text-muted-foreground">
                {result.slices.length} output slice{result.slices.length > 1 ? "s" : ""}
              </span>
            )}
          </div>

          {multi && (
            <SliceTabs slices={result.slices} active={sliceIndex} onSelect={setSliceIndex} />
          )}

          <IndicesChart result={result} sliceIndex={sliceIndex} fileName={fileName} />
          <ResultsTable result={result} sliceIndex={sliceIndex} />

          {result.notes.length > 0 && (
            <p className="text-xs text-muted-foreground">{result.notes.join(" ")}</p>
          )}
          {ref && <p className="font-mono text-[11px] text-muted-foreground">{ref}</p>}

          <div className="flex flex-wrap items-center justify-end gap-2 border-t border-border/60 pt-3">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() =>
                downloadText(`${fileName}-results.csv`, resultToCsv(result), "text/csv")
              }
              className="font-mono text-xs"
            >
              results csv
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

export function ResultsPanel({
  problem,
  designs,
  yData,
  results,
  activeDemo,
  onClear,
}: {
  problem: ProblemSpec | null;
  designs: Partial<Record<DesignMethod, GeneratedDesign>>;
  yData: YData | null;
  results: AnalysisResult[];
  activeDemo: Demo | null;
  onClear: () => void;
}) {
  const onSaveSession = () => {
    if (!problem) return;
    downloadText(
      "jaxgsa-session.json",
      JSON.stringify(buildSessionJson(problem, designs, yData, results), null, 2),
      "application/json",
    );
  };

  return (
    <div className="space-y-4">
      {results.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-sm text-muted-foreground">
            {results.length} result{results.length > 1 ? "s" : ""} · session bundle
            includes the problem, designs, outputs and all results
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <Button type="button" variant="outline" size="sm" onClick={onSaveSession}>
              Download session JSON
            </Button>
            <Button type="button" variant="ghost" size="sm" onClick={onClear}>
              clear
            </Button>
          </div>
        </div>
      )}

      <AnimatePresence initial={false}>
        {results.map((r, i) => (
          <ResultCard key={`${r.method}-${i}`} result={r} demo={activeDemo} index={i} />
        ))}
      </AnimatePresence>

      {results.length === 0 && (
        <Card className="border-dashed bg-card/45">
          <CardContent className="py-12 sm:py-16">
            <div className="mx-auto max-w-md text-center">
              <div className="mx-auto mb-5 flex size-12 items-center justify-center rounded-full border border-border bg-muted/35 font-mono text-sm text-muted-foreground">
                03
              </div>
              <p className="text-base text-foreground">Your analysis will appear here</p>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                Load a demo for a quick tour, or attach your own model outputs
                in Analyze. Results include an interactive chart, exact values,
                and reproducible downloads.
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
