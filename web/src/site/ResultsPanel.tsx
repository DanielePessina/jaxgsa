import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
} from "@/components/ui/card";
import { demoForProblem } from "./demos";
import { downloadText } from "./download";
import {
  buildSessionJson,
  resultToCsv,
  type AnalysisResult,
  type DesignMethod,
  type GeneratedDesign,
} from "./engine";
import { ResultsTable } from "./ResultsTable";
import type { ProblemSpec } from "@/jaxgsa/sampling";

function referenceLine(
  result: AnalysisResult,
  problem: ProblemSpec | null,
): string | null {
  if (result.method === "morris") return null;
  const demo = demoForProblem(problem);
  if (!demo) return null;
  const s1 = demo.reference.S1.map((v) => v.toFixed(3)).join(", ");
  const st = demo.reference.ST.map((v) => v.toFixed(3)).join(", ");
  return `reference (${demo.label}, analytical): S1 ≈ ${s1} · ST ≈ ${st}`;
}

export function ResultsPanel({
  problem,
  designs,
  yData,
  results,
  onClear,
}: {
  problem: ProblemSpec | null;
  designs: Partial<Record<DesignMethod, GeneratedDesign>>;
  yData: { values: Float64Array; label: string } | null;
  results: AnalysisResult[];
  onClear: () => void;
}) {
  if (results.length === 0) {
    return (
      <Card>
        <CardContent className="py-10 text-center text-sm text-muted-foreground">
          No results yet — run an analysis in section 03. Each method card
          there sends its result here.
        </CardContent>
      </Card>
    );
  }

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
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="font-mono text-xs text-muted-foreground">
          {results.length} result{results.length > 1 ? "s" : ""} · the session
          bundle includes the problem, designs, outputs and all results
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

      {results.map((r, i) => {
        const ref = referenceLine(r, problem);
        return (
          <div key={`${r.method}-${i}`} className="space-y-1.5">
            <ResultsTable result={r} />
            {ref && (
              <p className="px-1 font-mono text-[11px] text-muted-foreground">
                {ref}
              </p>
            )}
            <div className="flex justify-end">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() =>
                  downloadText(
                    `jaxgsa-${r.method}-results.csv`,
                    resultToCsv(r),
                    "text/csv",
                  )
                }
              >
                Download {r.method} CSV
              </Button>
            </div>
          </div>
        );
      })}
    </div>
  );
}