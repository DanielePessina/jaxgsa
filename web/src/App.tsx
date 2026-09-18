import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { AnalyzePanel } from "@/site/AnalyzePanel";
import { ProblemPanel } from "@/site/ProblemPanel";
import { ResultsPanel } from "@/site/ResultsPanel";
import { SamplePanel } from "@/site/SamplePanel";
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
import { demoForProblem, type Demo } from "@/site/demos";
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

type StepId = "problem" | "sample" | "analyze" | "results";

interface StepState {
  id: StepId;
  title: string;
  done: boolean;
  active: boolean;
}

function scrollTo(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

/** Compact status rail over the four stages of the workflow. */
function StepRail({ steps }: { steps: StepState[] }) {
  return (
    <nav aria-label="workflow steps" className="overflow-x-auto">
      <ol className="flex min-w-max items-center gap-1">
        {steps.map((s, i) => (
          <li key={s.id} className="flex items-center gap-1">
            {i > 0 && <span className="mx-1 text-muted-foreground/30">/</span>}
            <button
              type="button"
              onClick={() => scrollTo(s.id)}
              aria-current={s.active ? "step" : undefined}
              aria-label={`${s.title}: ${s.done ? "complete" : s.active ? "current" : "not started"}`}
              className={`relative flex items-center gap-2 rounded-full px-3 py-1.5 text-xs transition-colors ${
                s.active
                  ? "text-foreground"
                  : s.done
                    ? "text-muted-foreground hover:text-foreground"
                    : "text-muted-foreground/50 hover:text-muted-foreground"
              }`}
            >
              {s.active && (
                <motion.span
                  layoutId="step-rail-pill"
                  className="absolute inset-0 rounded-full bg-accent"
                  transition={{ type: "spring", stiffness: 380, damping: 34 }}
                />
              )}
              <span className="relative font-mono text-[10px]">
                {s.done ? "✓" : String(i + 1).padStart(2, "0")}
              </span>
              <span className="relative">{s.title}</span>
            </button>
          </li>
        ))}
      </ol>
    </nav>
  );
}

function Section({
  id,
  step,
  title,
  description,
  children,
}: {
  id: StepId;
  step: number;
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);
  return (
    <section id={id} className="scroll-mt-6">
      <div className="mb-4">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="group flex w-full items-baseline gap-3 text-left"
        >
          <span className="font-mono text-xs tabular-nums text-muted-foreground/60">
            {String(step).padStart(2, "0")}
          </span>
          <span className="flex-1">
            <span className="block text-lg font-normal tracking-tight text-foreground transition-colors group-hover:text-primary">
              {title}
            </span>
            {open && (
              <span className="mt-0.5 block max-w-2xl text-sm leading-relaxed text-muted-foreground">
                {description}
              </span>
            )}
          </span>
          <span className="font-mono text-xs text-muted-foreground/60">
            {open ? "—" : "+"}
          </span>
        </button>
        <div className="mt-2 border-t border-border/60" />
      </div>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ type: "spring", stiffness: 320, damping: 34 }}
            className="overflow-hidden"
          >
            <div className="pb-2">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
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
        baseN: 64,
        calcSecondOrder: false,
        seed: 0,
      });
      const y = demo.evaluate(gen);
      setDesigns({ sobol: gen });
      setYData(scalarYData(y, `demo model (${demo.label}) · ${y.length} runs`));
      setXSource({ kind: "design", method: "sobol" });
    } catch (err) {
      // Sampling failed (engine not ready yet, etc.): leave X unset; the
      // user can still generate a design in section 02.
      setXSource(null);
    }
  }, []);

  const sampleDone = xSource !== null;
  const analyzeDone = yData !== null;
  const resultsDone = results.length > 0;
  const activeStep: StepId =
    problem === null
      ? "problem"
      : !sampleDone
        ? "sample"
        : !analyzeDone
          ? "analyze"
          : "results";
  const steps: StepState[] = [
    { id: "problem", title: "Problem", done: problem !== null, active: activeStep === "problem" },
    { id: "sample", title: "Data", done: sampleDone, active: activeStep === "sample" },
    { id: "analyze", title: "Analyze", done: analyzeDone, active: activeStep === "analyze" },
    { id: "results", title: "Results", done: resultsDone, active: activeStep === "results" },
  ];

  return (
    <div className="min-h-screen">
      <header className="border-b border-border/60 bg-background/85 backdrop-blur">
        <div className="mx-auto flex max-w-[1560px] flex-wrap items-center justify-between gap-4 px-5 py-4 sm:px-8">
          <a href="#" className="flex items-baseline gap-3" aria-label="jaxgsa workbench home">
            <span className="text-xl font-semibold tracking-tight">jaxgsa</span>
            <span className="hidden text-sm text-muted-foreground sm:inline">browser workbench</span>
          </a>
          <div className="flex items-center gap-4">
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

      <main className="mx-auto max-w-[1560px] px-5 py-8 sm:px-8 lg:py-10">
        <div className="mb-9 grid gap-6 border-b border-border/60 pb-8 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
          <div>
            <p className="mb-3 font-mono text-xs uppercase tracking-[0.18em] text-primary">
              Global sensitivity analysis · local compute
            </p>
            <h1 className="max-w-4xl text-3xl font-medium tracking-[-0.035em] sm:text-4xl lg:text-5xl">
              Find which inputs move your model.
            </h1>
            <p className="mt-4 max-w-3xl text-base leading-relaxed text-muted-foreground">
              Define the input space, create an evaluation design, then bring
              back your model outputs. Sampling, analysis, and result export
              all run in this browser.
            </p>
          </div>
          <div className="flex flex-col items-start gap-2 lg:items-end">
            <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
              Python package
            </span>
            <div className="flex flex-wrap items-center gap-2">
              <InstallCommand command="pip install jaxgsa" />
              <InstallCommand command="uv add jaxgsa" />
            </div>
          </div>
        </div>

        <div className="sticky top-0 z-20 -mx-5 mb-8 border-y border-border/60 bg-background/90 px-5 py-3 backdrop-blur sm:-mx-8 sm:px-8 lg:static lg:mx-0 lg:rounded-lg lg:border lg:bg-card/35 lg:px-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <StepRail steps={steps} />
            <div className="hidden items-center gap-4 font-mono text-[11px] text-muted-foreground md:flex">
              <span>files stay local</span>
              <span className="text-border">·</span>
              <span>{device?.webgpuAvailable ? "WebGPU available · using WASM f64" : "WASM float64"}</span>
            </div>
          </div>
        </div>

        <div className="xl:grid xl:grid-cols-[minmax(620px,1fr)_minmax(500px,0.82fr)] xl:items-start xl:gap-10 2xl:gap-14">
          <div className="min-w-0 space-y-12">
            <Section
              id="problem"
              step={1}
              title="Problem"
              description="Every analysis starts with the inputs of your model: name each one and give it a distribution. Your model stays on your machine — we only ever sample from these distributions and read the outputs you bring back."
            >
              <ProblemPanel
                problem={problem}
                onProblemChange={onProblemChange}
                onDemoChange={setActiveDemo}
                onLoadDemo={onLoadDemo}
              />
            </Section>

            <Section
              id="sample"
              step={2}
              title="Choose data"
              description="Start a new Sobol or Morris experiment, or continue with input-output data you already have."
            >
              <SamplePanel
                problem={problem}
                designs={designs}
                onDesign={onDesign}
                onUseExisting={() => scrollTo("analyze")}
              />
            </Section>

            <Section
              id="analyze"
              step={3}
              title="Analyze"
              description="Attach the outputs and run the methods. Sampled a design? Upload the matching Y, joined by run_id. No design? Upload your own X and Y — PCE and Shapley work on any point cloud."
            >
              <AnalyzePanel
                problem={problem}
                designs={designs}
                yData={yData}
                setYData={setYData}
                appendResult={appendResult}
                xSource={xSource}
                setXSource={setXSource}
                givenX={givenX}
                setGivenX={setGivenX}
                givenXLabel={givenXLabel}
                setGivenXLabel={setGivenXLabel}
                activeDemo={activeDemo}
                onGoSample={() => scrollTo("sample")}
                onGoResults={() => scrollTo("results")}
              />
            </Section>
          </div>

          <aside
            id="results"
            className="mt-12 min-w-0 scroll-mt-20 border-t border-border/60 pt-8 xl:sticky xl:top-6 xl:mt-0 xl:max-h-[calc(100vh-3rem)] xl:overflow-y-auto xl:border-l xl:border-t-0 xl:pl-8 xl:pr-2 xl:pt-0 2xl:pl-10"
          >
            <div className="mb-5 flex items-baseline justify-between gap-4">
              <div>
                <p className="font-mono text-xs uppercase tracking-[0.16em] text-muted-foreground">Output</p>
                <p className="mt-1 text-2xl font-medium tracking-tight">Results</p>
              </div>
              <span className="text-xs text-muted-foreground">
                {results.length === 0 ? "waiting for an analysis" : `${results.length} saved in this session`}
              </span>
            </div>
            <ResultsPanel
              problem={problem}
              designs={designs}
              yData={yData}
              results={results}
              activeDemo={activeDemo}
              onClear={() => setResults([])}
            />
          </aside>
        </div>
      </main>

      <footer className="border-t border-border/60">
        <div className="mx-auto flex max-w-[1560px] flex-wrap items-center justify-between gap-3 px-5 py-5 text-sm text-muted-foreground sm:px-8">
          <span>Runs entirely in your browser — nothing leaves this page.</span>
          <span className="font-mono text-xs">jaxgsa 0.9.1 reference</span>
        </div>
      </footer>
    </div>
  );
}
