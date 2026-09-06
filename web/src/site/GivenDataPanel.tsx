import { useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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
import { ResultsTable } from "./ResultsTable";
import {
  ISHIGAMI_PROBLEM,
  analyzeCloud,
  loadExampleCloud,
  parseCsv,
  type AnalysisResult,
  type GivenDataMethod,
} from "./engine";

const METHOD_BLURB: Record<GivenDataMethod, string> = {
  pce: "Polynomial chaos expansion on an arbitrary (X, Y) cloud — S1, ST from the fitted coefficients.",
  shapley: "Shapley effects aggregated from a PCE fit — Sh, plus the S1/ST bounds of the same expansion.",
};

export function GivenDataPanel({ method }: { method: GivenDataMethod }) {
  const [order, setOrder] = useState("3");
  const [combined, setCombined] = useState(false);
  const [x, setX] = useState<Float64Array | null>(null);
  const [y, setY] = useState<Float64Array | null>(null);
  const [xLabel, setXLabel] = useState("no X data loaded");
  const [yLabel, setYLabel] = useState("no Y data loaded");
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const D = ISHIGAMI_PROBLEM.names.length;

  const onExample = () => {
    setError(null);
    const { x: xv, y: yv, n } = loadExampleCloud(1024);
    setX(xv);
    setY(yv);
    setXLabel(`Ishigami cloud · ${n} rows × ${D} cols`);
    setYLabel(`Ishigami cloud · ${n} outputs`);
    setResults(null);
  };

  const onX = (parsed: ReturnType<typeof parseCsv>, label: string) => {
    setError(null);
    try {
      if (parsed.rows.length === 0) throw new Error("the X CSV has no data rows");
      const bad = parsed.rows.find((r) => r.length !== D);
      if (bad !== undefined) {
        throw new Error(
          `the X CSV must have exactly ${D} columns per row (this problem has ${D} parameters)`,
        );
      }
      const flat = new Float64Array(parsed.rows.length * D);
      for (let i = 0; i < parsed.rows.length; i++) {
        for (let j = 0; j < D; j++) flat[i * D + j] = parsed.rows[i][j];
      }
      setX(flat);
      setXLabel(`${label} · ${parsed.rows.length} rows × ${D} cols`);
      setResults(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const onY = (parsed: ReturnType<typeof parseCsv>, label: string) => {
    setError(null);
    try {
      const flat = new Float64Array(parsed.rows.length);
      for (let i = 0; i < parsed.rows.length; i++) {
        if (parsed.rows[i].length < 1) {
          throw new Error(`line ${i + 2}: row has no output value`);
        }
        flat[i] = parsed.rows[i][0];
      }
      setY(flat);
      setYLabel(`${label} · ${flat.length} outputs`);
      setResults(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const onCombined = (parsed: ReturnType<typeof parseCsv>) => {
    setError(null);
    try {
      if (parsed.rows.length === 0) throw new Error("the combined CSV has no data rows");
      const cols = parsed.headers.length;
      if (cols < D + 1) {
        throw new Error(
          `the combined CSV needs ${D} X columns + 1 Y column, found ${cols}`,
        );
      }
      const bad = parsed.rows.find((r) => r.length < D + 1);
      if (bad !== undefined) {
        throw new Error("a combined row has fewer columns than D + 1");
      }
      const flatX = new Float64Array(parsed.rows.length * D);
      const flatY = new Float64Array(parsed.rows.length);
      for (let i = 0; i < parsed.rows.length; i++) {
        for (let j = 0; j < D; j++) flatX[i * D + j] = parsed.rows[i][j];
        flatY[i] = parsed.rows[i][parsed.rows[i].length - 1];
      }
      setX(flatX);
      setY(flatY);
      setXLabel(`combined · ${parsed.rows.length} rows`);
      setYLabel(`combined · last column as Y`);
      setResults(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const onRun = async () => {
    if (!x || !y) return;
    setError(null);
    setBusy(true);
    await new Promise((r) => setTimeout(r, 10));
    try {
      setResults(analyzeCloud(method, ISHIGAMI_PROBLEM, x, y, Number(order)));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
            {method} — configuration
          </CardTitle>
          <CardDescription>{METHOD_BLURB[method]}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-4">
            <div className="space-y-1.5">
              <Label htmlFor="order" className="font-mono text-xs">
                order
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
            <Button type="button" variant="outline" onClick={onExample}>
              Load example (Ishigami cloud)
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
            data — N×{D} point cloud
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">input format</span>
            <Button
              type="button"
              size="sm"
              variant={combined ? "ghost" : "default"}
              onClick={() => setCombined(false)}
            >
              separate X + Y files
            </Button>
            <Button
              type="button"
              size="sm"
              variant={combined ? "default" : "ghost"}
              onClick={() => setCombined(true)}
            >
              one combined CSV
            </Button>
          </div>

          {combined ? (
            <CsvUpload
              label="Upload combined CSV"
              hint={`first ${D} columns = X (${ISHIGAMI_PROBLEM.names.join(", ")}), last column = Y`}
              disabled={busy}
              onLoaded={onCombined}
              onError={setError}
            />
          ) : (
            <>
              <CsvUpload
                label="Upload X CSV"
                hint={`${D} columns (x1, x2, x3), N rows`}
                disabled={busy}
                onLoaded={(p) => onX(p, "uploaded X")}
                onError={setError}
              />
              <CsvUpload
                label="Upload Y CSV"
                hint="one output column, N rows"
                disabled={busy}
                onLoaded={(p) => onY(p, "uploaded Y")}
                onError={setError}
              />
            </>
          )}

          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="font-mono text-xs text-muted-foreground">
              {xLabel} · {yLabel}
            </p>
            <Button type="button" disabled={!x || !y || busy} onClick={() => void onRun()}>
              {busy ? "computing…" : "Run analysis"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {results && <ResultsTable result={results} />}

      {error && (
        <Alert variant="destructive">
          <AlertTitle>engine error</AlertTitle>
          <AlertDescription className="font-mono">{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}