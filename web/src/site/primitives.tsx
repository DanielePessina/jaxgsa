/**
 * Small shared UI pieces used across the sections: error banner, mono
 * select, per-name chips, and the compact number field.
 */

import { useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { ProblemSpec } from "@/jaxgsa/sampling";

export function ErrorBanner({
  title,
  message,
}: {
  title: string;
  message: string;
}) {
  const [copied, setCopied] = useState(false);
  return (
    <Alert variant="destructive">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <AlertTitle>{title}</AlertTitle>
          <AlertDescription className="mt-1 break-words font-mono text-xs leading-relaxed">
            {message}
          </AlertDescription>
        </div>
        <button
          type="button"
          onClick={() => {
            void navigator.clipboard.writeText(message);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}
          className="shrink-0 cursor-pointer rounded-sm border border-border px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground transition-colors hover:border-destructive/50 hover:text-foreground"
          title="copy error message"
        >
          {copied ? "copied ✓" : "copy"}
        </button>
      </div>
    </Alert>
  );
}

/** Hand-styled mono select used for the demo picker and the X-source picker. */
export function MonoSelect({
  value,
  onChange,
  children,
  className = "",
}: {
  value: string;
  onChange: (v: string) => void;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`rounded-sm border border-input bg-background px-2 py-1 font-mono text-xs ${className}`}
    >
      {children}
    </select>
  );
}

const CHIP_CLASS =
  "rounded-sm border border-border bg-muted/40 px-2 py-0.5 font-mono text-xs text-muted-foreground";

/** "D = N" badge plus one chip per parameter name. */
export function ProblemChips({ problem }: { problem: ProblemSpec }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Badge variant="outline" className="font-mono text-xs">
        D = {problem.names.length}
      </Badge>
      {problem.names.map((n) => (
        <span key={n} className={CHIP_CLASS}>
          {n}
        </span>
      ))}
    </div>
  );
}

/** Compact labeled number input (mono). */
export function NumberField({
  id,
  label,
  value,
  onChange,
  min,
  step = 1,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  min?: number;
  step?: number;
}) {
  return (
    <div className="space-y-1">
      <Label htmlFor={id} className="font-mono text-xs">
        {label}
      </Label>
      <Input
        id={id}
        type="number"
        min={min}
        step={step}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-8 font-mono text-xs"
      />
    </div>
  );
}