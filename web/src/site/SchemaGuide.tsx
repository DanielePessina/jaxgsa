import type { ReactNode } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ProblemSpec } from "@/jaxgsa/sampling";

/**
 * Renders the X/Y CSV contract as small sample tables so a first-time user
 * can match their file to what the uploaders accept. Two variants: the design
 * route (run_id-aligned) and the uploaded-data route (row order).
 */

const SAMPLE_X = [
  ["-1.5942", "2.3018", "0.0082"],
  ["2.8136", "-0.9481", "1.5694"],
  ["-0.2234", "1.7186", "-2.0911"],
];

function SampleTable({
  caption,
  headers,
  rows,
  muted,
}: {
  caption: string;
  headers: string[];
  rows: string[][];
  muted?: boolean[];
}) {
  return (
    <Table>
      <caption className="sr-only">{caption}</caption>
      <TableHeader>
        <TableRow>
          {headers.map((h, j) => (
            <TableHead
              key={h}
              className={`font-mono text-xs ${muted?.[j] ? "text-muted-foreground/60" : ""}`}
            >
              {h}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((row, i) => (
          <TableRow key={i}>
            {row.map((cell, j) => (
              <TableCell
                key={j}
                className={`font-mono text-xs tabular-nums ${muted?.[j] ? "text-muted-foreground/50" : ""}`}
              >
                {cell}
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <p className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
      {children}
    </p>
  );
}

export function SchemaGuide({
  problem,
  route,
  method,
}: {
  problem: ProblemSpec;
  route: "design" | "uploaded" | "none";
  method?: string;
}) {
  const names = problem.names;
  const D = names.length;
  const output = problem.outputNames?.[0] ?? "y";
  const yTimeCols = [output, `${output}_t0.5`, `${output}_t1`];

  return (
    <div className="rounded-sm border border-border/60 bg-muted/20 p-3">
      <p className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
        expected schema
      </p>
      {route === "none" ? (
        <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
          Load X first — the file format depends on your X source. A sampled
          design uses a <span className="font-mono text-foreground">run_id</span>{" "}
          join key; your own uploads match by row order.
        </p>
      ) : route === "design" ? (
        <div className="mt-2 space-y-3">
          <div className="space-y-1">
            <SectionLabel>X · {method} design CSV</SectionLabel>
            <p className="text-xs leading-relaxed text-muted-foreground">
              The design from section 02, in{" "}
              <span className="font-mono text-foreground">run_id</span> order.
              Use it as-is or download the CSV.
            </p>
            <SampleTable
              caption="X design CSV: run_id followed by one column per parameter"
              headers={["run_id", ...names]}
              rows={[
                ["0", ...SAMPLE_X[0].slice(0, D)],
                ["1", ...SAMPLE_X[1].slice(0, D)],
                ["2", ...SAMPLE_X[2].slice(0, D)],
              ]}
            />
          </div>
          <div className="space-y-1">
            <SectionLabel>Y · run_id-aligned</SectionLabel>
            <p className="text-xs leading-relaxed text-muted-foreground">
              One output column per model output.{" "}
              <span className="font-mono text-foreground">run_id</span> is the
              join key — it may be in{" "}
              <span className="font-mono text-foreground">any row order</span>{" "}
              and need not start at 0. Parameter columns are ignored, so you can
              append outputs to the downloaded design CSV.
            </p>
            <SampleTable
              caption="Y CSV: run_id in any order, output columns appended"
              headers={["run_id", ...names.slice(0, 1), output]}
              muted={[false, true, false]}
              rows={[
                ["2", SAMPLE_X[2][0], "0.98"],
                ["0", SAMPLE_X[0][0], "0.33"],
                ["1", SAMPLE_X[1][0], "-0.71"],
              ]}
            />
          </div>
          <div className="space-y-1">
            <SectionLabel>Y · time-resolved</SectionLabel>
            <p className="text-xs leading-relaxed text-muted-foreground">
              Time slices are named{" "}
              <span className="font-mono text-foreground">
                {yTimeCols.join(", ")}
              </span>
              , … — any finite decimal works ({`y_t0.25`}, {`y_t3`}).
            </p>
            <SampleTable
              caption="Time-resolved Y CSV: one column per time slice"
              headers={["run_id", ...yTimeCols]}
              rows={[
                ["0", "0.33", "0.29", "0.12"],
                ["1", "-0.71", "-0.66", "-0.52"],
              ]}
            />
          </div>
        </div>
      ) : (
        <div className="mt-2 space-y-3">
          <div className="space-y-1">
            <SectionLabel>X · your upload</SectionLabel>
            <p className="text-xs leading-relaxed text-muted-foreground">
              Header is exactly the parameter names in order, one row per model
              run. No <span className="font-mono text-foreground">run_id</span>.
            </p>
            <SampleTable
              caption="X upload: header exactly the parameter names, in order"
              headers={names}
              rows={SAMPLE_X.slice(0, 3).map((r) => r.slice(0, D))}
            />
          </div>
          <div className="space-y-1">
            <SectionLabel>Y · your upload</SectionLabel>
            <p className="text-xs leading-relaxed text-muted-foreground">
              One output column per model output, one value per X row, in the{" "}
              <span className="font-mono text-foreground">same row order</span>.
              No <span className="font-mono text-foreground">run_id</span>{" "}
              needed.
            </p>
            <SampleTable
              caption="Y upload: one value per X row, in row order"
              headers={[output]}
              rows={[["0.33"], ["-0.71"], ["0.98"]]}
            />
          </div>
          <div className="space-y-1">
            <SectionLabel>Y · time-resolved</SectionLabel>
            <p className="text-xs leading-relaxed text-muted-foreground">
              Time slices are named{" "}
              <span className="font-mono text-foreground">
                {yTimeCols.join(", ")}
              </span>
              , … — one column per time slice, same row order.
            </p>
            <SampleTable
              caption="Time-resolved Y upload: one column per time slice"
              headers={yTimeCols}
              rows={[
                ["0.33", "0.29", "0.12"],
                ["-0.71", "-0.66", "-0.52"],
              ]}
            />
          </div>
        </div>
      )}
    </div>
  );
}