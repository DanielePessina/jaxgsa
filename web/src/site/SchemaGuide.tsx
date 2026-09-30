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
 * Renders the X/Y CSV contract as file cards: an explicit column layout and
 * row layout statement per file, with a small sample table under each. Two
 * variants: the design route (run_id-aligned) and the uploaded-data route
 * (row order).
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

function Meta({ children }: { children: ReactNode }) {
  return (
    <p className="text-xs leading-relaxed text-muted-foreground">{children}</p>
  );
}

function Mono({ children }: { children: ReactNode }) {
  return (
    <span className="font-mono text-foreground">{children}</span>
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <p className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
      {children}
    </p>
  );
}

/** One expected-file card: name, explicit column layout, explicit row layout. */
function FileCard({
  label,
  title,
  columns,
  rows,
  extra,
  table,
}: {
  label: string;
  title: string;
  /** Explicit column list rendered in mono. */
  columns: ReactNode;
  /** Explicit row contract sentence. */
  rows: ReactNode;
  extra?: ReactNode;
  table: ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <SectionLabel>
        {label} · {title}
      </SectionLabel>
      <div className="space-y-0.5 rounded-sm border border-border/60 bg-background/60 px-2.5 py-2">
        <p className="font-mono text-xs">
          <span className="text-muted-foreground">columns: </span>
          {columns}
        </p>
        <p className="font-mono text-xs leading-relaxed">
          <span className="text-muted-foreground">rows: </span>
          {rows}
        </p>
      </div>
      {extra}
      {table}
    </div>
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
          Load a data route first — the X and Y file layouts depend on it. A
          sampled design joins output files with a{" "}
          <Mono>run_id</Mono> column; your own uploads match by row order.
        </p>
      ) : route === "design" ? (
        <div className="mt-2 space-y-4">
          <FileCard
            label="X file"
            title={`${method ?? "design"} design CSV`}
            columns={
              <>
                <Mono>run_id</Mono>
                {names.map((n) => (
                  <span key={n}>
                    , <Mono>{n}</Mono>
                  </span>
                ))}
              </>
            }
            rows="one per design point; run_id counts the rows 0…N−1 in order"
            table={
              <SampleTable
                caption="X design CSV: run_id followed by one column per parameter"
                headers={["run_id", ...names]}
                rows={[
                  ["0", ...SAMPLE_X[0].slice(0, D)],
                  ["1", ...SAMPLE_X[1].slice(0, D)],
                  ["2", ...SAMPLE_X[2].slice(0, D)],
                ]}
              />
            }
          />
          <FileCard
            label="Y file (inputs sampled here)"
            title="outputs aligned by run_id"
            columns={
              <>
                <Mono>run_id</Mono> then one or more output columns
                {names.slice(0, 1).length > 0 && (
                  <>
                    {" "}
                    (parameter columns like <Mono>{names[0]}</Mono> are ignored)
                  </>
                )}
              </>
            }
            rows="one per design row, in ANY row order — run_id says which design row each value belongs to"
            extra={
              <Meta>
                Each output column must cover every design row exactly once.
              </Meta>
            }
            table={
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
            }
          />
          <FileCard
            label="Y file variant"
            title="time-resolved outputs"
            columns={
              <>
                one column per time slice:{" "}
                {yTimeCols.map((c, j) => (
                  <span key={c}>
                    {j > 0 && ", "}
                    <Mono>{c}</Mono>
                  </span>
                ))}
              </>
            }
            rows="same as above — one row per design row, aligned by run_id; any row order"
            extra={
              <Meta>
                <Mono>{output}</Mono> alone means time 0; any finite decimal
                time works (<Mono>y_t0.25</Mono>, <Mono>y_t3</Mono>). An output
                literally named <Mono>{output}_t2</Mono> must be written{" "}
                <Mono>{output}_t2_t0</Mono>.
              </Meta>
            }
            table={
              <SampleTable
                caption="Time-resolved Y CSV: one column per time slice"
                headers={["run_id", ...yTimeCols]}
                rows={[
                  ["0", "0.33", "0.29", "0.12"],
                  ["1", "-0.71", "-0.66", "-0.52"],
                ]}
              />
            }
          />
        </div>
      ) : (
        <div className="mt-2 space-y-4">
          <FileCard
            label="X file"
            title="your upload"
            columns={
              names.length > 0 ? (
                <>
                  exactly{" "}
                  {names.map((n, j) => (
                    <span key={n}>
                      {j > 0 && ", "}
                      <Mono>{n}</Mono>
                    </span>
                  ))}{" "}
                  in this order — no <Mono>run_id</Mono>
                </>
              ) : null
            }
            rows="one per model run"
            extra={
              D === 0 ? (
                <Meta>Define at least one input in step 01.</Meta>
              ) : undefined
            }
            table={
              <SampleTable
                caption="X upload: header exactly the parameter names, in order"
                headers={names}
                rows={SAMPLE_X.slice(0, 3).map((r) => r.slice(0, D))}
              />
            }
          />
          <FileCard
            label="Y file (X uploaded)"
            title="outputs by row order"
            columns={
              <>
                one output column per output slice: <Mono>{output}</Mono>
                {D > 0 && <> — no <Mono>run_id</Mono></>}
              </>
            }
            rows={
              <>
                one per X row, in the <strong>same row order</strong> as the X
                file
              </>
            }
            extra={
              <Meta>
                Row 3 of the Y file is the output of row 3 of the X file — the
                values are matched by position.
              </Meta>
            }
            table={
              <SampleTable
                caption="Y upload: one value per X row, in row order"
                headers={[output]}
                rows={[["0.33"], ["-0.71"], ["0.98"]]}
              />
            }
          />
          <FileCard
            label="Y file variant"
            title="time-resolved outputs"
            columns={
              <>
                one column per time slice:{" "}
                {yTimeCols.map((c, j) => (
                  <span key={c}>
                    {j > 0 && ", "}
                    <Mono>{c}</Mono>
                  </span>
                ))}
              </>
            }
            rows="one per X row, in the same row order as the X file"
            table={
              <SampleTable
                caption="Time-resolved Y upload: one column per time slice"
                headers={yTimeCols}
                rows={[
                  ["0.33", "0.29", "0.12"],
                  ["-0.71", "-0.66", "-0.52"],
                ]}
              />
            }
          />
        </div>
      )}
    </div>
  );
}
