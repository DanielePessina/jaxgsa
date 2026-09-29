import { useCallback, useEffect, useRef, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AnalyzePanel } from "@/site/AnalyzePanel";
import { DataPanel, dataRouteStatus, type DataRoute } from "@/site/DataPanel";
import { ProblemPanel } from "@/site/ProblemPanel";
import { ResultsPanel } from "@/site/ResultsPanel";
import { StageNav } from "@/site/StageNav";
import { ThemeToggle } from "@/site/ThemeToggle";
import {
  generateDesign,
  initEngine,
  scalarYData,
  type AnalysisResult,
  type DesignMethod,
  type DeviceInfo,
  type GeneratedDesign,
  type XSource,
  type YData,
} from "@/site/engine";
import { demoById, demoForProblem, type Demo } from "@/site/demos";
import type { ProblemSpec } from "@/jaxgsa/sampling";

function InstallCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <button
      type="button"
      onClick={() => {
        void navigator.clipboard.writeText(command);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
      className="cursor-pointer rounded-sm border border-border bg-muted/40 px-2 py-0.5 font-mono text-xs text-foreground transition-colors hover:border-primary/50"
      title="click to copy"
    >
      {copied ? "copied ✓" : command}
    </button>
  );
}

type StageId = "problem" | "data" | "analyze" | "results";

const STAGES: { id: StageId; label: string }[] = [
  { id: "problem", label: "Problem" },
  { id: "data", label: "Data" },
  { id: "analyze", label: "Analyze" },
  { id: "results", label: "Results" },
];

/** One stage section: heading row (always visible) + body when active. */
function StageSection({
  index,
  title,
  status,
  done,
  active,
  locked,
  lockedNote,
  onGo,
  next,
  children,
}: {
  index: number;
  title: string;
  status: string;
  done: boolean;
  active: boolean;
  locked: boolean;
  lockedNote: string | null;
  onGo: () => void;
  next: { label: string; disabled: boolean; hint?: string; onClick: () => void } | null;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);

  return (
    <section id={`stage-${STAGES[index].id}`} className="scroll-mt-24">
      <button
        type="button"
        onClick={() => active ? setOpen((o) => !o) : onGo()}
        disabled={locked}
        aria-expanded={active ? open : undefined}
        className="group flex w-full items-baseline gap-3 text-left"
      >
        <span
          className={`mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full border font-mono text-[10px] tabular-nums ${
            done
              ? "border-primary/50 bg-primary/10 text-primary"
              : locked
                ? "border-border text-muted-foreground/50"
                : active
                  ? "border-primary/60 bg-primary/10 text-primary"
                  : "border-border text-muted-foreground"
          }`}
        >
          {done ? "✓" : String(index + 1).padStart(2, "0")}
        </span>
        <span className="flex-1">
          <span
            className={`block text-lg font-normal tracking-tight transition-colors ${
              locked ? "text-muted-foreground/60" : "text-foreground group-hover:text-primary"
            }`}
          >
            {title}
          </span>
        </span>
        <span className={`font-mono text-[11px] ${done ? "text-primary" : "text-muted-foreground"}`}>
          {status}
        </span>
      </button>
      <div className="mt-1.5 border-t border-border/60" />

      {locked ? (
        <p className="pl-8 pt-3 text-xs text-muted-foreground">{lockedNote}</p>
      ) : active ? (
        open ? (
          <div className="pt-4">
            {children}
            {next && (
              <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-border/60 pt-3">
                <span className="font-mono text-[11px] text-muted-foreground">
                  {next.hint ?? ""}
                </span>
                <Button size="sm" disabled={next.disabled} onClick={next.onClick}>
                  {next.label} →
                </Button>
              </div>
            )}
          </div>
        ) : (
          <p className="pl-8 pt-3 text-xs text-muted-foreground">
            collapsed — click the title to reopen
          </p>
        )
      ) : null}
    </section>
  );
}

export default function App() {
  const [device, setDevice] = useState<DeviceInfo | null>(null);
  const [initError, setInitError] = useState<string | null>(null);
  const [problem, setProblem] = useState<ProblemSpec | null>(null);

  const [designs, setDesigns] = useState<
    Partial<Record<DesignMethod, GeneratedDesign>>
  >({});

  const [yData, setYData] = useState<YData | null>(null);
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [xSource, setXSource] = useState<XSource | null>(null);
  const [givenX, setGivenX] = useState<Float64Array | null>(null);
  const [givenXLabel, setGivenXLabel] = useState("no X data loaded");
  const [activeDemo, setActiveDemo] = useState<Demo | null>(null);
  const [route, setRoute] = useState<DataRoute>({ kind: "new" });

  const [stage, setStage] = useState<StageId>("problem");

  // The demo problem the ProblemPanel's table should mirror after a hero CTA
  // load; ProblemPanel applies it when the counter changes.
  const [demoRequest, setDemoRequest] = useState<{ demo: Demo } | null>(null);

  // Guards the "rows effect" re-fires from ProblemPanel: when it reports a
  // problem we already hold, nothing resets.
  const problemRef = useRef<ProblemSpec | null>(null);

  useEffect(() => {
    let cancelled = false;
    initEngine()
      .then((d) => {
        if (!cancelled) setDevice(d);
      })
      .catch((err) => {
        if (!cancelled) setInitError(err instanceof Error ? err.message : String(err));
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const onProblemChange = useCallback((p: ProblemSpec) => {
    if (problemRef.current && JSON.stringify(problemRef.current) === JSON.stringify(p)) {
      return;
    }

    problemRef.current = p;
    setProblem(p);
    setDesigns({});
    setYData(null);
    setResults([]);
    setXSource(null);
    setGivenX(null);
    setGivenXLabel("no X data loaded");
    setActiveDemo(demoForProblem(p));
  }, []);

  const onDesign = useCallback((gen: GeneratedDesign) => {
    setDesigns((d) => ({ ...d, [gen.method]: gen }));
    setYData(null);
    setResults([]);
    setXSource({ kind: "design", method: gen.method });
  }, []);

  const appendResult = useCallback((r: AnalysisResult) => {
    setResults((rs) => [...rs, r]);
  }, []);

  const onYDataChange = useCallback((next: YData | null) => {
    setYData(next);
    setResults([]);
  }, []);

  /**
   * One-click demo pipeline (PLAN-WEB-UI D5b): load the demo's problem,
   * sample a Sobol' design in the wasm runtime, evaluate the demo model on
   * it, and wire the whole thing as the Analyze X source — ready to run.
   */
  const onLoadDemo = useCallback((demo: Demo) => {
    problemRef.current = demo.problem;
    setProblem(demo.problem);
    setActiveDemo(demo);
    setDesigns({});
    setYData(null);
    setResults([]);
    setGivenX(null);
    setGivenXLabel("no X data loaded");
    setXSource(null);

    try {
      const gen = generateDesign("sobol", demo.problem, {
        baseN: 256,
        calcSecondOrder: false,
        seed: 0,
      });

      const y = demo.evaluate(gen);
      setDesigns({ sobol: gen });
      setYData(scalarYData(y, `demo model (${demo.label}) · ${y.length} runs`));
      setXSource({ kind: "design", method: "sobol" });
    } catch {
      // Sampling failed (engine not ready yet, etc.): leave X unset; the
      // user can still generate a design in the data stage.
      setXSource(null);
    }
  }, []);

  /** Hero CTA: load the Ishigami demo and land the user on the analysis step. */
  const onStartDemo = useCallback(() => {
    const demo = demoById("ishigami");
    onLoadDemo(demo);
    setRoute({ kind: "new" });
    setDemoRequest({ demo });
    setStage("analyze");
  }, [onLoadDemo]);

  const problemDone = problem !== null;
  const dataReady = dataRouteStatus(xSource, yData).ready;
  const dataDone = xSource !== null && yData !== null;
  const resultsDone = results.length > 0;

  const lockedFor = (index: number): boolean => {
    switch (STAGES[index].id) {
      case "problem":
        return false;
      case "data":
        return !problemDone;
      case "analyze":
        return !(problemDone && xSource !== null);
      case "results":
        return !(problemDone && xSource !== null && yData !== null);
    }
  };

  const navStages = STAGES.map((s, i) => ({
    id: s.id,
    label: s.label,
    done:
      s.id === "problem"
        ? problemDone
        : s.id === "data"
          ? dataDone
          : s.id === "analyze"
            ? resultsDone
            : resultsDone,
    locked: lockedFor(i),
  }));

  const activeIndex = STAGES.findIndex((s) => s.id === stage);

  const goStage = (id: StageId) => setStage(id);

  return (
    <div className="min-h-screen">
      <header className="border-b border-border/60 bg-background/85 backdrop-blur">
        <div className="mx-auto flex max-w-[1560px] flex-wrap items-center justify-between gap-4 px-5 py-4 sm:px-8">
          <a href="#" className="flex items-baseline gap-3" aria-label="jaxgsa workbench home">
            <span className="text-xl font-semibold tracking-tight">jaxgsa</span>
            <span className="hidden text-sm text-muted-foreground sm:inline">browser workbench</span>
          </a>
          <div className="flex items-center gap-4">
            <ThemeToggle />
            <a
              href="https://danielepessina.github.io/jaxgsa"
              target="_blank"
              rel="noreferrer"
              className="text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              Documentation ↗
            </a>
            <Badge variant="outline" className="font-mono text-xs">
              WASM · f64
            </Badge>
          </div>
        </div>
      </header>

      {initError && (
        <div className="mx-auto max-w-[1560px] px-5 pt-4 sm:px-8">
          <Alert variant="destructive">
            <AlertTitle>engine failed to start</AlertTitle>
            <AlertDescription className="font-mono">{initError}</AlertDescription>
          </Alert>
        </div>
      )}

      <main className="mx-auto max-w-3xl px-5 py-8 sm:px-8 lg:py-10">
        <div className="mb-9 border-b border-border/60 pb-8">
          <p className="mb-3 font-mono text-xs uppercase tracking-[0.18em] text-primary">
            Global sensitivity analysis · local compute
          </p>
          <h1 className="text-3xl font-medium tracking-[-0.035em] sm:text-4xl">
            Find which inputs move your model.
          </h1>
          <p className="mt-4 max-w-2xl text-base leading-relaxed text-muted-foreground">
            Define the input space, create an evaluation design, then bring
            back your model outputs. Sampling, analysis, and result export
            all run in this browser — four steps, nothing leaves this page.
          </p>
          <div className="mt-5 flex flex-wrap items-center gap-3">
            <Button
              type="button"
              onClick={() => {
                if (device === null) return;
                onStartDemo();
              }}
              disabled={device === null}
            >
              Run the 2-minute demo
            </Button>
            <span className="text-xs text-muted-foreground">
              full workflow: problem → design → outputs → indices
            </span>
          </div>
        </div>

        <div className="sticky top-0 z-20 -mx-5 mb-8 border-y border-border/60 bg-background/90 px-5 py-3 backdrop-blur sm:-mx-8 sm:px-8">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <StageNav stages={navStages} active={activeIndex} onGo={(i) => goStage(STAGES[i].id)} />
            <div className="hidden items-center gap-4 font-mono text-[11px] text-muted-foreground md:flex">
              <span>files stay local</span>
              <span className="text-border">·</span>
              <span>
                {device?.webgpuAvailable ? "WebGPU available · using WASM f64" : "WASM float64"}
              </span>
            </div>
          </div>
        </div>

        <div className="space-y-12">
          <StageSection
            index={0}
            title="Problem — the inputs of your model"
            status={problemDone ? `D = ${problem?.names.length ?? 0}` : "not defined"}
            done={problemDone}
            active={stage === "problem"}
            locked={false}
            lockedNote={null}
            onGo={() => goStage("problem")}
            next={{
              label: "Define the data",
              disabled: !problemDone,
              hint: problemDone ? undefined : "name each input and give it a distribution first",
              onClick: () => goStage("data"),
            }}
          >
            <ProblemPanel
              problem={problem}
              onProblemChange={onProblemChange}
              onDemoChange={setActiveDemo}
              onLoadDemo={(d) => {
                onLoadDemo(d);
                setRoute({ kind: "new" });
              }}
              demoRequest={demoRequest}
            />
          </StageSection>

          <StageSection
            index={1}
            title="Data — sample a design or bring your own"
            status={dataRouteStatus(xSource, yData).summary}
            done={dataDone}
            active={stage === "data"}
            locked={!problemDone}
            lockedNote="Define the problem in step 01 first."
            onGo={() => goStage("data")}
            next={{
              label: "Choose the methods",
              disabled: !dataReady,
              hint: dataDone
                ? undefined
                : xSource === null
                  ? "generate a design or upload X first"
                  : "attach the model outputs (Y) to unlock the methods",
              onClick: () => goStage("analyze"),
            }}
          >
            {problem && (
              <DataPanel
                problem={problem}
                designs={designs}
                route={route}
                onRouteChange={setRoute}
                onDesign={onDesign}
                xSource={xSource}
                setXSource={setXSource}
                yData={yData}
                onYData={onYDataChange}
                givenX={givenX}
                setGivenX={setGivenX}
                givenXLabel={givenXLabel}
                setGivenXLabel={setGivenXLabel}
                activeDemo={activeDemo}
                disabled={false}
              />
            )}
          </StageSection>

          <StageSection
            index={2}
            title="Analyze"
            status={resultsDone ? `${results.length} result${results.length === 1 ? "" : "s"}` : "waiting"}
            done={resultsDone}
            active={stage === "analyze"}
            locked={!(problemDone && xSource !== null)}
            lockedNote="Generate a design or upload X in step 02 first."
            onGo={() => goStage("analyze")}
            next={{ label: "View results", disabled: !resultsDone, onClick: () => goStage("results") }}
          >
            {problem && (
              <AnalyzePanel
                problem={problem}
                designs={designs}
                yData={yData}
                appendResult={appendResult}
                xSource={xSource}
                setXSource={setXSource}
                givenX={givenX}
                onGoResults={() => goStage("results")}
              />
            )}
          </StageSection>

          <StageSection
            index={3}
            title="Results"
            status={resultsDone ? `${results.length} in this session` : "empty"}
            done={resultsDone}
            active={stage === "results"}
            locked={!(problemDone && xSource !== null && yData !== null)}
            lockedNote="Run an analysis in step 03 first."
            onGo={() => goStage("results")}
            next={{
              label: "Adjust the analysis",
              disabled: false,
              onClick: () => goStage("analyze"),
            }}
          >
            {dataReady && (
              <ResultsPanel
                problem={problem}
                designs={designs}
                yData={yData}
                results={results}
                activeDemo={activeDemo}
                onClear={() => setResults([])}
              />
            )}
          </StageSection>
        </div>
      </main>

      <footer className="border-t border-border/60">
        <div className="mx-auto flex max-w-[1560px] flex-wrap items-center justify-between gap-3 px-5 py-5 text-sm text-muted-foreground sm:px-8">
          <span className="flex flex-wrap items-center gap-3">
            <span>Runs entirely in your browser — nothing leaves this page.</span>
            <InstallCommand command="pip install jaxgsa" />
            <InstallCommand command="uv add jaxgsa" />
          </span>
          <span className="font-mono text-xs">jaxgsa 0.9.1 reference</span>
        </div>
      </footer>
    </div>
  );
}
