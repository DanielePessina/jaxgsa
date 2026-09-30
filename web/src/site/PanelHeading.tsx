import type { ReactNode } from "react";

/**
 * Uniform panel heading: normal-case sentence for the instructional landmark
 * the user reads to learn the flow. Mono/uppercase stays reserved for small
 * machine labels (kicker line), matching the workbench's existing voice.
 */
export function PanelHeading({
  kicker,
  title,
  sub,
}: {
  kicker: string;
  title: string;
  sub?: ReactNode;
}) {
  return (
    <div>
      <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
        {kicker}
      </p>
      <h2 className="mt-1 text-lg font-normal tracking-tight text-foreground">
        {title}
      </h2>
      {sub && (
        <div className="mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
          {sub}
        </div>
      )}
    </div>
  );
}
