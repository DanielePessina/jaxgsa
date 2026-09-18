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
    <nav aria-label="workflow steps">
      <ol className="flex flex-wrap items-center gap-1">
        {steps.map((s, i) => (
          <li key={s.id} className="flex items-center gap-1">
            {i > 0 && <span className="mx-1 text-muted-foreground/40">→</span>}
            <button
              type="button"
              onClick={() => scrollTo(s.id)}
              aria-current={s.active ? "step" : undefined}
              aria-label={`${s.title}: ${s.done ? "complete" : s.active ? "current" : "not started"}`}
              className={`relative flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs transition-colors ${
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
              <span className="relative">{s.done ? "✓" : "·"}</span>
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
    { id: "sample", title: "Sample", done: sampleDone, active: activeStep === "sample" },
    { id: "analyze", title: "Analyze", done: analyzeDone, active: activeStep === "analyze" },
    { id: "results", title: "Results", done: resultsDone, active: activeStep === "results" },
  ];

  return (
    <div className="min-h-screen">
      <header className="border-b border-border/60">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-6 py-5">
          <div>
            <h1 className="text-xl font-medium tracking-tight">
              jaxgsa
              <span className="ml-3 text-sm font-normal text-muted-foreground">
                global sensitivity analysis, in your browser
              </span>
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="font-mono text-xs">
              wasm · float64
            </Badge>
          </div>
        </div>
      </header>

      <div className="border-b border-border/60 bg-muted/10">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-3 gap-y-2 px-6 py-2.5 text-sm">
          <span className="text-muted-foreground">Full package:</span>
          <InstallCommand command="pip install jaxgsa" />
          <span className="text-muted-foreground/50">or</span>
          <InstallCommand command="uv add jaxgsa" />
          <a
            href="https://danielepessina.github.io/jaxgsa"
            target="_blank"
            rel="noreferrer"
            className="ml-1 text-sm underline decoration-dotted underline-offset-4 hover:text-foreground"
          >
            docs
          </a>
        </div>
      </div>

      {device?.webgpuAvailable && (
        <div className="mx-auto max-w-6xl px-6 pt-4">
          <Alert>
            <AlertTitle className="font-mono text-xs uppercase tracking-wider">
              WebGPU detected
            </AlertTitle>
            <AlertDescription>
              GPU acceleration is not wired up yet — running on wasm float64
              for exactness.
            </AlertDescription>
          </Alert>
        </div>
      )}

      {initError && (
        <div className="mx-auto max-w-6xl px-6 pt-4">
          <Alert variant="destructive">
            <AlertTitle>engine failed to start</AlertTitle>
            <AlertDescription className="font-mono">{initError}</AlertDescription>
          </Alert>
        </div>
      )}

      <main className="mx-auto max-w-6xl px-6 py-8">
        <p className="mb-8 max-w-3xl text-sm leading-relaxed text-muted-foreground">
          Global sensitivity analysis (GSA) shows how much each input of your
          model contributes to the variability of its output. Define your
          inputs, draw a design, run your own model, upload the outputs — the
          indices compute entirely in your browser.
        </p>

        <div className="mb-6">
          <StepRail steps={steps} />
        </div>

        <div className="xl:grid xl:grid-cols-[minmax(0,1fr)_400px] xl:items-start xl:gap-8">
          <div className="space-y-10">
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
              title="Sample"
              description="Generate the input designs. Download the CSV, evaluate it with your model, and bring the outputs back to the Analyze section."
            >
              <SamplePanel problem={problem} designs={designs} onDesign={onDesign} />
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
            className="mt-10 scroll-mt-6 xl:sticky xl:top-6 xl:mt-0 xl:max-h-[calc(100vh-3rem)] xl:overflow-y-auto xl:pr-1"
          >
            <div className="mb-4">
              <p className="text-lg font-normal tracking-tight">Results</p>
              <div className="mt-2 border-t border-border/60" />
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
        <div className="mx-auto max-w-6xl px-6 py-5 text-sm text-muted-foreground">
          Runs entirely in your browser — nothing leaves this page.
        </div>
      </footer>
    </div>
  );
}
