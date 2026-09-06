import { useCallback, useEffect, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { AnalyzePanel } from "@/site/AnalyzePanel";
import { ProblemPanel } from "@/site/ProblemPanel";
import { ResultsPanel } from "@/site/ResultsPanel";
import { SamplePanel } from "@/site/SamplePanel";
import {
  initEngine,
  type AnalysisResult,
  type DesignMethod,
  type DeviceInfo,
  type GeneratedDesign,
} from "@/site/engine";
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

function Section({
  id,
  number,
  title,
  description,
  children,
}: {
  id: string;
  number: string;
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-6">
      <div className="mb-3 border-b border-border pb-2">
        <h2 className="font-mono text-sm font-semibold uppercase tracking-widest">
          {number} · {title}
        </h2>
        <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
          {description}
        </p>
      </div>
      {children}
    </section>
  );
}

function scrollTo(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
}

export default function App() {
  const [device, setDevice] = useState<DeviceInfo | null>(null);
  const [initError, setInitError] = useState<string | null>(null);
  const [problem, setProblem] = useState<ProblemSpec | null>(null);
  const [designs, setDesigns] = useState<
    Partial<Record<DesignMethod, GeneratedDesign>>
  >({});
  const [yData, setYData] = useState<{
    values: Float64Array;
    label: string;
  } | null>(null);
  const [results, setResults] = useState<AnalysisResult[]>([]);

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
    setProblem(p);
    setDesigns({});
    setYData(null);
    setResults([]);
  }, []);

  const onDesign = useCallback((gen: GeneratedDesign) => {
    setDesigns((d) => ({ ...d, [gen.method]: gen }));
    setYData(null);
    setResults([]);
  }, []);

  const appendResult = useCallback((r: AnalysisResult) => {
    setResults((rs) => [...rs, r]);
  }, []);

  return (
    <div className="min-h-screen">
      <header className="border-b border-border">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-6 py-4">
          <div>
            <h1 className="font-mono text-lg font-semibold tracking-tight">
              jaxgsa
              <span className="ml-2 font-sans text-sm font-normal text-muted-foreground">
                Global sensitivity analysis, in your browser
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

      <div className="border-b border-border bg-muted/20">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-3 gap-y-2 px-6 py-2.5 text-sm">
          <span className="font-medium">Like it? Install the full package:</span>
          <InstallCommand command="pip install jaxgsa" />
          <span className="text-muted-foreground/60">or</span>
          <InstallCommand command="uv add jaxgsa" />
          <a
            href="https://danielepessina.github.io/jaxgsa"
            target="_blank"
            rel="noreferrer"
            className="ml-1 underline decoration-dotted underline-offset-4 hover:text-foreground"
          >
            docs
          </a>
        </div>
      </div>

      <div className="mx-auto max-w-6xl px-6 pt-6">
        <p className="max-w-3xl text-sm leading-relaxed text-muted-foreground">
          Global sensitivity analysis (GSA) shows how much each input of your
          model contributes to the variability of its output. Define your
          inputs below, draw a design, run your own model, upload the outputs —
          the indices compute entirely in your browser.
        </p>
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

      <main className="mx-auto max-w-6xl space-y-10 px-6 py-8">
        <Section
          id="problem"
          number="01"
          title="Problem"
          description="Every analysis starts with the inputs of your model: name each one and give it a distribution. Your model stays on your machine — we only ever sample from these distributions and read the outputs you bring back."
        >
          <ProblemPanel problem={problem} onProblemChange={onProblemChange} />
        </Section>

        <Section
          id="sample"
          number="02"
          title="Sample"
          description="Generate the input designs. Download the CSV, evaluate it with your model, and bring the outputs back to the Analyze section."
        >
          <SamplePanel problem={problem} designs={designs} onDesign={onDesign} />
        </Section>

        <Section
          id="analyze"
          number="03"
          title="Analyze"
          description="Attach the outputs and run the methods. Sampled a design? Upload the matching Y, joined by run_id. No design? Upload your own X and Y — PCE and Shapley work on any point cloud."
        >
          <AnalyzePanel
            problem={problem}
            designs={designs}
            yData={yData}
            setYData={setYData}
            appendResult={appendResult}
            onGoSample={() => scrollTo("sample")}
            onGoResults={() => scrollTo("results")}
          />
        </Section>

        <Section
          id="results"
          number="04"
          title="Results"
          description="Inspect the indices, compare them with the analytical references of the demo problems, and save. The session JSON bundles the whole run — problem, designs, outputs and results."
        >
          <ResultsPanel
            problem={problem}
            designs={designs}
            yData={yData}
            results={results}
            onClear={() => setResults([])}
          />
        </Section>
      </main>

      <footer className="border-t border-border">
        <div className="mx-auto max-w-6xl px-6 py-5 text-sm text-muted-foreground">
          Runs entirely in your browser — nothing leaves this page.
        </div>
      </footer>
    </div>
  );
}