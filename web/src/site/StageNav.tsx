import { motion } from "motion/react";

/**
 * The workflow stepper: one button per stage. Locked stages (upstream data
 * still missing) stay visible but disabled so the user can always see the
 * whole path.
 */
export interface NavStage {
  id: string;
  label: string;
  done: boolean;
  locked: boolean;
}

export function StageNav({
  stages,
  active,
  onGo,
}: {
  stages: NavStage[];
  active: number;
  onGo: (i: number) => void;
}) {
  return (
    <nav aria-label="workflow steps">
      <ol className="flex flex-wrap items-center gap-x-0.5 gap-y-1 sm:gap-x-1">
        {stages.map((s, i) => (
          <li key={s.id} className="flex items-center gap-1">
            {i > 0 && <span className="mx-1 hidden text-muted-foreground/30 sm:inline">/</span>}
            <button
              type="button"
              disabled={s.locked}
              title={
                s.locked
                  ? "finish the earlier steps first"
                  : i === active
                    ? "current step"
                    : s.done
                      ? "done — click to revisit"
                      : "go back to this step"
              }
              aria-current={i === active ? "step" : undefined}
              onClick={() => onGo(i)}
              className={`relative flex items-center gap-1.5 rounded-full px-2.5 py-1.5 sm:gap-2 sm:px-3 text-xs transition-colors ${
                s.locked
                  ? "cursor-not-allowed text-muted-foreground/40"
                  : i === active
                    ? "text-foreground"
                    : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {i === active && (
                <motion.span
                  layoutId="stage-nav-pill"
                  className="absolute inset-0 rounded-full bg-accent"
                  transition={{ type: "spring", stiffness: 380, damping: 34 }}
                />
              )}
              <span className="relative font-mono text-[10px]">
                {s.done ? "✓" : String(i + 1).padStart(2, "0")}
              </span>
              <span className="relative">{s.label}</span>
            </button>
          </li>
        ))}
      </ol>
    </nav>
  );
}
