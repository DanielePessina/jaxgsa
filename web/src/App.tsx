import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AnalyzePanel } from "@/site/AnalyzePanel";
import { DataPanel, dataRouteStatus, type DataRoute } from "@/site/DataPanel";
import { ProblemPanel } from "@/site/ProblemPanel";
import { ResultsPanel } from "@/site/ResultsPanel";
import { StageNav } from "@/site/StageNav";
import { ThemeToggle } from "@/site/ThemeToggle";
import { clearPersistedSession, usePersistentState } from "@/site/session";
import {
  analyzeCloud,
  analyzeGenerated,
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
import { demoById, DEMOS, type Demo, type DemoId } from "@/site/demos";
import {
  clampStage,
  defaultRows,
  demoStillMatches,
  isProblemRows,
  problemKey,
  referenceDemo,
  validateProblem,
  rowsFromProblem,
  type ProblemRow,
} from "@/site/problemRows";
import { version } from "../package.json";
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

/** Methods the hero demo runs on the Ishigami design, in this order. */
const DEMO_METHODS = ["sobol", "pce", "shapley", "borgonovo", "pawn"] as const;

/** PCE and Shapley order for the demo. Order 3 is too low for Ishigami. */
const DEMO_ORDER = 8;

/** Base sample size of the demo's Sobol' design. */
const DEMO_BASE_N = 256;

const tick = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

/**
 * One stage section: heading row (always visible) + body. The body stays
 * mounted while hidden, so opening or collapsing a section never resets the
 * panel inside it.
 */
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
  const id = STAGES[index].id;

  // Going to a section always opens it.
  useEffect(() => {
    if (active) setOpen(true);
  }, [active]);

  return (
    <section id={`stage-${id}`} className="scroll-mt-24">
      <button
        type="button"
        id={`stage-heading-${id}`}
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

      {locked && <p className="pl-8 pt-3 text-xs text-muted-foreground">{lockedNote}</p>}
      {!locked && active && !open && (
        <p className="pl-8 pt-3 text-xs text-muted-foreground">
          collapsed — click the title to reopen
        </p>
      )}

      <div hidden={locked || !active || !open} className="pt-4">
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
    </section>
  );
}

function demoIdOrDefault(value: string): DemoId {
  const hit = DEMOS.find((d) => d.id === value);

  return hit ? hit.id : "linear";
}

export default function App() {
  const [device, setDevice] = useState<DeviceInfo | null>(null);
  const [initError, setInitError] = useState<string | null>(null);

  // Autosaved: the problem table, output names, the dropdown demo and the stage.
  // Uploaded data, designs and results are never saved.
  const [rowsStored, setRows] = usePersistentState<ProblemRow[]>("rows", defaultRows());
  const [outputNamesRaw, setOutputNamesRaw] = usePersistentState("output-names", "");
  const [demoIdStored, setDemoId] = usePersistentState<string>("demo-id", "linear");
  const [stageStored, setStageStored] = usePersistentState<string>("stage", "problem");

  const rows = useMemo(() => {
    try {
      return isProblemRows(rowsStored) ? rowsStored : defaultRows();
    } catch {
      return defaultRows();
    }
  }, [rowsStored]);

  const demoId = demoIdOrDefault(demoIdStored);

  const validation = useMemo(() => {
    try {
      return validateProblem(rows, outputNamesRaw);
    } catch {
      return validateProblem(defaultRows(), "");
    }
  }, [rows, outputNamesRaw]);

  // The problem the designs, Y and results belong to. It changes only when
  // the table holds a valid problem that differs from this one.
  const [problem, setProblem] = useState<ProblemSpec | null>(() => validation.problem);

  const [designs, setDesigns] = useState<
    Partial<Record<DesignMethod, GeneratedDesign>>
  >({});

  const [yData, setYData] = useState<YData | null>(null);
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [xSource, setXSource] = useState<XSource | null>(null);
  const [givenX, setGivenX] = useState<Float64Array | null>(null);
  const [givenXLabel, setGivenXLabel] = useState("no X data loaded");

  // The demo that was actually loaded. Cleared once the problem differs from it.
  const [activeDemo, setActiveDemo] = useState<Demo | null>(null);
  const [route, setRoute] = useState<DataRoute>({ kind: "new" });
  const [demoError, setDemoError] = useState<string | null>(null);
  const [demoRunning, setDemoRunning] = useState<string | null>(null);

  // Changing the key remounts every panel, which drops their local state.
  const [sessionKey, setSessionKey] = useState(0);

  // Key of the problem the downstream data belongs to.
  const problemRef = useRef<string | null>(problem ? problemKey(problem) : null);

  // Bumped on every problem change or reset, so a running demo can tell it is stale.
  const runToken = useRef(0);

  // The stage that should take focus after the next stage change.
  const focusStage = useRef<StageId | null>(null);

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

  const clearDownstream = useCallback(() => {
    runToken.current += 1;
    setDesigns({});
    setYData(null);
    setResults([]);
    setXSource(null);
    setGivenX(null);
    setGivenXLabel("no X data loaded");
  }, []);

  // `validation` changes identity only on a real edit of the table or of the
  // output names. A valid problem that differs from the current one clears the
  // data built on the old problem.
  useEffect(() => {
    const next = validation.problem;

    if (next === null) return;
    const key = problemKey(next);

    if (key === problemRef.current) return;

    problemRef.current = key;
    runToken.current += 1;
    setDemoRunning(null);
    setProblem(next);
    clearDownstream();
    setActiveDemo((demo) => demoStillMatches(demo, next));
  }, [validation, clearDownstream]);

  const onDesign = useCallback((gen: GeneratedDesign) => {
    runToken.current += 1;
    setDemoRunning(null);
    setDesigns((d) => ({ ...d, [gen.method]: gen }));
    setYData(null);
    setResults([]);
    setXSource({ kind: "design", method: gen.method });
  }, []);

  const appendResult = useCallback((r: AnalysisResult) => {
    setResults((rs) => [...rs, r]);
  }, []);

  const onYDataChange = useCallback((next: YData | null) => {
    runToken.current += 1;
    setDemoRunning(null);
    setYData(next);
    setResults([]);
  }, []);

  /** Put a demo's problem into the table and the app, and drop old data. */
  const applyDemoProblem = useCallback(
    (demo: Demo) => {
      runToken.current += 1;
      setDemoRunning(null);
      problemRef.current = problemKey(demo.problem);
      setRows(rowsFromProblem(demo.problem));
      setOutputNamesRaw("");
      setDemoId(demo.id);
      setProblem(demo.problem);
      setActiveDemo(demo);
      clearDownstream();
      setDemoError(null);
    },
    [setRows, setOutputNamesRaw, setDemoId, clearDownstream],
  );

  /** Sample the demo's Sobol' design and evaluate the demo model on it. */
  const sampleDemo = useCallback((demo: Demo) => {
    const gen = generateDesign("sobol", demo.problem, {
      baseN: DEMO_BASE_N,
      calcSecondOrder: false,
      seed: 0,
    });

    const y = demo.evaluate(gen);
    const data = scalarYData(y, `demo model (${demo.label}) · ${y.length} runs`);

    setDesigns({ sobol: gen });
    setYData(data);
    setXSource({ kind: "design", method: "sobol" });

    return { gen, data };
  }, []);

  /** "Start Sobol demo": load the demo's problem, design and Y. Show failures. */
  const onLoadDemo = useCallback(
    (demo: Demo) => {
      applyDemoProblem(demo);
      setRoute({ kind: "new" });

      try {
        sampleDemo(demo);
      } catch (err) {
        setDemoError(`Could not load the ${demo.label} demo: ${err instanceof Error ? err.message : String(err)}`);
      }
    },
    [applyDemoProblem, sampleDemo],
  );

  const goStage = useCallback(
    (id: StageId, options?: { focus?: boolean }) => {
      focusStage.current = options?.focus ? id : null;
      setStageStored(id);
    },
    [setStageStored],
  );

  /**
   * Hero CTA: load Ishigami, sample, evaluate, run every method that fits
   * this design, and land on Results. Compute is synchronous, so it yields to
   * paint between methods.
   */
  const onStartDemo = useCallback(async () => {
    const demo = demoById("ishigami");
    applyDemoProblem(demo);
    setRoute({ kind: "new" });
    const token = runToken.current;
    const stale = () => runToken.current !== token;

    try {
      setDemoRunning("Sampling the design");
      await tick();

      if (stale()) return;
      const { gen, data } = sampleDemo(demo);

      for (const method of DEMO_METHODS) {
        setDemoRunning(`Running ${method}`);
        await tick();

        if (stale()) return;

        const result =
          method === "sobol"
            ? await analyzeGenerated(gen, data)
            : analyzeCloud(method, demo.problem, gen.samples, data, DEMO_ORDER);

        if (stale()) return;
        appendResult(result);
      }

      goStage("results", { focus: true });
    } catch (err) {
      if (!stale()) {
        setDemoError(`The demo failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    } finally {
      if (!stale()) setDemoRunning(null);
    }
  }, [applyDemoProblem, sampleDemo, appendResult, goStage]);

  const onResetSession = () => {
    if (!window.confirm("Reset the session? This clears the problem, data and results.")) return;

    clearPersistedSession();
    runToken.current += 1;
    const fresh = defaultRows();
    const freshProblem = validateProblem(fresh, "").problem;
    problemRef.current = freshProblem ? problemKey(freshProblem) : null;
    setRows(fresh);
    setOutputNamesRaw("");
    setDemoId("linear");
    setProblem(freshProblem);
    setActiveDemo(null);
    clearDownstream();
    setRoute({ kind: "new" });
    setDemoError(null);
    setDemoRunning(null);
    setStageStored("problem");
    setSessionKey((k) => k + 1);
  };

  const tableValid = validation.problem !== null && problem !== null;
  const problemDone = tableValid;
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

  const lockedFlags = STAGES.map((_, i) => lockedFor(i));
  const wantedIndex = STAGES.findIndex((s) => s.id === stageStored);
  const activeIndex = clampStage(wantedIndex, lockedFlags);
  const stage = STAGES[activeIndex].id;

  // After a reload only the problem is restored, so a saved later stage
  // falls back to the furthest one that is unlocked.
  useEffect(() => {
    if (stage !== stageStored) setStageStored(stage);
  }, [stage, stageStored, setStageStored]);

  // Read by async handlers that must act only while a stage is still shown.
  const stageRef = useRef<StageId>(stage);
  stageRef.current = stage;

  // A "next" button moves focus to the heading of the section it opened.
  useEffect(() => {
    if (focusStage.current !== stage) return;
    focusStage.current = null;
    document.getElementById(`stage-heading-${stage}`)?.focus();
  }, [stage]);

  const navStages = STAGES.map((s, i) => ({
    id: s.id,
    label: s.label,
    done:
      s.id === "problem"
        ? problemDone
        : s.id === "data"
          ? dataDone
          : resultsDone,
    locked: lockedFlags[i],
  }));

  // The analytical reference shows only for Y made by the demo model.
  const resultsDemo = referenceDemo(activeDemo, problem, yData?.label ?? null);

  return (
    <div className="min-h-screen">
      <header className="border-b border-border/60 bg-background/85 backdrop-blur">
        <div className="mx-auto flex max-w-[1560px] flex-wrap items-center justify-between gap-4 px-5 py-4 sm:px-8">
          <a href="#" className="flex items-baseline gap-3" aria-label="jaxgsa studio home">
            <span className="text-xl font-semibold tracking-tight">jaxgsa</span>
            <span className="text-sm text-muted-foreground">studio</span>
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
                void onStartDemo();
              }}
              disabled={device === null || demoRunning !== null}
            >
              {demoRunning === null ? "Run the demo" : "Running the demo…"}
            </Button>
            <span className="text-xs text-muted-foreground" role="status">
              {demoRunning ?? "full workflow: problem → design → outputs → indices"}
            </span>
          </div>
        </div>

        <div className="sticky top-0 z-20 -mx-5 mb-8 border-y border-border/60 bg-background/90 px-5 py-3 backdrop-blur sm:-mx-8 sm:px-8">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <StageNav
              stages={navStages}
              active={activeIndex}
              onGo={(i) => goStage(STAGES[i].id)}
            />
            <div className="flex items-center gap-4 font-mono text-[11px] text-muted-foreground">
              <span className="hidden items-center gap-4 md:flex">
                <span>files stay local</span>
                <span className="text-border">·</span>
                <span>
                  {device?.webgpuAvailable ? "WebGPU available · using WASM f64" : "WASM float64"}
                </span>
              </span>
              <button
                type="button"
                onClick={onResetSession}
                className="cursor-pointer underline-offset-2 transition-colors hover:text-foreground hover:underline"
              >
                Reset session
              </button>
            </div>
          </div>
        </div>

        {demoError && (
          <Alert variant="destructive" className="mb-8">
            <AlertTitle>demo error</AlertTitle>
            <AlertDescription className="font-mono">{demoError}</AlertDescription>
          </Alert>
        )}

        <div key={sessionKey} className="space-y-12">
          <StageSection
            index={0}
            title="Problem — the inputs of your model"
            status={problemDone ? `D = ${problem?.names.length ?? 0}` : "needs a fix"}
            done={problemDone}
            active={stage === "problem"}
            locked={false}
            lockedNote={null}
            onGo={() => goStage("problem")}
            next={{
              label: "Define the data",
              disabled: !problemDone,
              hint: validation.issue === null ? undefined : `Fix this first: ${validation.issue}`,
              onClick: () => goStage("data", { focus: true }),
            }}
          >
            <ProblemPanel
              problem={problem}
              rows={rows}
              onRowsChange={setRows}
              outputNamesRaw={outputNamesRaw}
              onOutputNamesChange={setOutputNamesRaw}
              demoId={demoId}
              onDemoIdChange={setDemoId}
              validation={validation}
              onLoadDemo={onLoadDemo}
            />
          </StageSection>

          <StageSection
            index={1}
            title="Data — sample a design or bring your own"
            status={dataRouteStatus(xSource, yData).summary}
            done={dataDone}
            active={stage === "data"}
            locked={!problemDone}
            lockedNote={
              validation.issue === null
                ? "Define the problem in step 01 first."
                : `Fix the problem in step 01 first: ${validation.issue}`
            }
            onGo={() => goStage("data")}
            next={{
              label: "Choose the methods",
              disabled: !dataReady,
              hint: dataDone
                ? undefined
                : xSource === null
                  ? "generate a design or upload X first"
                  : "attach the model outputs (Y) to unlock the methods",
              onClick: () => goStage("analyze", { focus: true }),
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
                activeDemo={demoStillMatches(activeDemo, problem)}
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
            locked={lockedFlags[2]}
            lockedNote="Generate a design or upload X in step 02 first."
            onGo={() => goStage("analyze")}
            next={{
              label: "View results",
              disabled: !resultsDone,
              onClick: () => goStage("results", { focus: true }),
            }}
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
                onGoResults={() => {
                  if (stageRef.current === "analyze") goStage("results", { focus: true });
                }}
              />
            )}
          </StageSection>

          <StageSection
            index={3}
            title="Results"
            status={resultsDone ? `${results.length} in this session` : "empty"}
            done={resultsDone}
            active={stage === "results"}
            locked={lockedFlags[3]}
            lockedNote="Run an analysis in step 03 first."
            onGo={() => goStage("results")}
            next={{
              label: "Adjust the analysis",
              disabled: false,
              onClick: () => goStage("analyze", { focus: true }),
            }}
          >
            {dataReady && (
              <ResultsPanel
                problem={problem}
                designs={designs}
                yData={yData}
                results={results}
                activeDemo={resultsDemo}
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
          <span className="flex items-center gap-3">
            <Badge variant="outline" className="font-mono text-xs">
              WASM · f64
            </Badge>
            <span className="font-mono text-xs">jaxgsa {version} reference</span>
          </span>
        </div>
      </footer>
    </div>
  );
}
