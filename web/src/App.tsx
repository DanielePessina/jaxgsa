import { useEffect, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { DesignMethodPanel } from "@/site/DesignMethodPanel";
import { GivenDataPanel } from "@/site/GivenDataPanel";
import { initEngine, type DeviceInfo } from "@/site/engine";

type MethodKey = "sobol" | "morris" | "kucherenko" | "pce" | "shapley" | "hdmr";

const DESIGN_METHODS: MethodKey[] = ["sobol", "morris", "kucherenko"];
const GIVEN_METHODS: MethodKey[] = ["pce", "shapley", "hdmr"];

function NavItem({
  label,
  active,
  disabled,
  onClick,
}: {
  label: string;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <Button
      type="button"
      variant={active ? "secondary" : "ghost"}
      disabled={disabled}
      onClick={onClick}
      className={`w-full justify-start font-mono text-sm ${disabled ? "cursor-not-allowed text-muted-foreground/60" : ""}`}
    >
      {label}
      {disabled && <span className="ml-auto text-[10px] text-muted-foreground/60">soon</span>}
    </Button>
  );
}

function HdmrPanel() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="font-mono text-sm uppercase tracking-wider text-muted-foreground">
          HDMR
        </CardTitle>
        <CardDescription>
          High-dimensional model representation — coming soon. The port needs
          control-flow (scan/backfit) and a scalar betainc before it can ship.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Badge variant="outline" className="font-mono text-xs">
          not yet ported
        </Badge>
      </CardContent>
    </Card>
  );
}

export default function App() {
  const [device, setDevice] = useState<DeviceInfo | null>(null);
  const [initError, setInitError] = useState<string | null>(null);
  const [method, setMethod] = useState<MethodKey>("sobol");

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

  return (
    <div className="min-h-screen">
      <header className="border-b border-border">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-6 py-4">
          <div>
            <h1 className="font-mono text-lg font-semibold tracking-tight">
              jaxgsa
              <span className="ml-2 font-sans text-sm font-normal text-muted-foreground">
                sensitivity analysis, in your browser
              </span>
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="font-mono text-xs">
              wasm · float64
            </Badge>
            {device?.webgpuAvailable && (
              <Badge className="font-mono text-xs">webgpu detected</Badge>
            )}
          </div>
        </div>
      </header>

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

      <main className="mx-auto flex max-w-6xl gap-8 px-6 py-8">
        <aside className="w-52 shrink-0">
          <nav className="space-y-4">
            <div>
              <p className="mb-1.5 px-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                design-based methods
              </p>
              {DESIGN_METHODS.map((m) => (
                <NavItem
                  key={m}
                  label={m}
                  active={method === m}
                  onClick={() => setMethod(m)}
                />
              ))}
            </div>
            <Separator />
            <div>
              <p className="mb-1.5 px-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                given data (surrogates)
              </p>
              {GIVEN_METHODS.map((m) => (
                <NavItem
                  key={m}
                  label={m}
                  active={method === m}
                  disabled={m === "hdmr"}
                  onClick={() => setMethod(m)}
                />
              ))}
            </div>
          </nav>
        </aside>

        <section className="min-w-0 flex-1">
          {method === "hdmr" ? (
            <HdmrPanel />
          ) : method === "pce" || method === "shapley" ? (
            <GivenDataPanel key={method} method={method} />
          ) : (
            <DesignMethodPanel key={method} method={method} />
          )}
        </section>
      </main>
    </div>
  );
}