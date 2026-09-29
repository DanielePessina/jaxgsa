---
name: jaxgsa-web
description: Use when working on the jaxgsa browser workbench under web/, especially the three-step problem → data → analysis flow, the X/Y CSV/Parquet upload contracts, run_id alignment, the expected-schema UI, and the site engine. Covers where each piece lives, what file formats and column layouts the uploaders accept, the exact error strings users can hit, and the tests that pin the contract.
---

# jaxgsa-web

The WASM/React workbench at `web/` runs ported sensitivity analyses locally in the
browser (float64, no data leaves the page). The user flow is four stages rendered
one at a time by the stepper: **01 Problem → 02 Data → 03 Analyze → 04 Results**
(`App.tsx` stage state + `StageNav`; data route readiness via `DataPanel.dataRouteStatus`).

```ts
web/src/App.tsx                 // stage state machine: stage 0..3, problem, designs, givenX, yData, results
web/src/site/StageNav.tsx       // stepper rail (done/locked states, locked = upstream data missing)
web/src/site/PanelHeading.tsx   // shared panel heading (kicker + title + sub)
web/src/site/ProblemPanel.tsx   // marginals + optional output names; accepts hero `demoRequest`
web/src/site/DataPanel.tsx      // ONE place for data: route cards (new design vs. uploads),
                                // design generation (ex-SamplePanel), X/Y uploads, SchemaGuide
web/src/site/AnalyzePanel.tsx   // method selection + run (no uploads); exports `availability()`
web/src/site/FileUpload.tsx     // drag-drop file input (CSV/Parquet)
web/src/site/upload.ts          // format sniffing, unsafe-format rejection, size guards
web/src/site/engine.ts          // parse/align/build helpers (the contract lives here)
web/src/site/SchemaGuide.tsx    // rendered CSV sample tables under "expected schema"
web/src/site/demos.ts           // loadable benchmarks (demo model evaluate)
web/src/site/ResultsPanel.tsx   // results cards, downloads, session JSON
web/src/jaxgsa/                 // ported method math (borgonovo, efast, pawn, ...)
```

## The four-stage flow

1. **01 Problem** — parameter names, marginals (uniform/gaussian/truncated-gaussian),
   optional comma-separated output names. `outputNames` mirror Python
   `Problem.output_names`; the default output is `y` when absent.
2. **02 Data** — one panel owning the data decision: two route cards
   ("Start a new experiment" → generate a Sobol/Morris/Kucherenko/eFAST design,
   download the CSV, upload Y with `run_id`; or "I have completed model runs" →
   upload X then Y positionally). The expected-schema guide renders below the
   uploads keyed off `xSource`. Ready = X source chosen + Y attached.
3. **03 Analyze** — pick methods (availability-aware) + run. No uploads here;
   X/Y summaries only.
4. **04 Results** — full-width results stage (no longer a sticky side dock).

## X/Y CSV contracts

### X upload ("my own X", given-data route)

- Header must be **exactly the problem parameter names, in order**.
- One numeric row per model run. **No `run_id` column.**
- Errors: `the X CSV has no data rows`, `the X CSV header must be exactly: <names>`,
  `line <n>: expected <D> columns, found <k>`.

### Y upload, design route (xSource.kind === "design")

- The Y file **must contain a `run_id` column**; otherwise:
  `the Y file must contain a "run_id" column matching the design download`.
- `run_id` must be a **permutation of `0..N-1`** (the design's canonical row order,
  i.e. the row index of the downloaded design CSV). Row order in Y is irrelevant —
  columns are re-aligned onto the design by `run_id`.
- Columns whose header matches a problem parameter name are **ignored** — so a user
  can append output columns to the downloaded design CSV and upload the whole file.
- `run_id` is also ignored as a data column (it is the join key).
- Errors: `run_id row count mismatch: the design has N rows but the uploaded Y CSV has M`,
  `run_id column mismatch: expected N ids, found M`,
  `line <n>: invalid run_id <v> (expected an integer in [0, N))`,
  `line <n>: duplicate run_id <v>`,
  `missing run_id <v>: the uploaded Y CSV must cover every design row`.

### Y upload, given-data route (uploaded X)

- **Positional**: one output column per model output, one value per X row, in the
  **same row order** as X. No `run_id` needed (a `run_id` column, if present, is
  extracted and ignored).
- Row count must equal X's row count, else `X has <N> rows but Y has <M> values`.

### Time-resolved outputs (both routes)

- Suffix `_t<number>` — any finite decimal, incl. signs/scientific notation:
  `y_t0`, `y_t0.5`, `y_t3`, `y_t-1`, `y_t2e1`.
- A bare name (no suffix) is that output at time 0. Within an output channel, time
  coordinates are sorted; channels keep first-seen header order.
- Escape hatch: an output literally named `foo_t2` must be written `foo_t2_t0`.
- Errors: `no output columns found: the Y file needs at least one column besides run_id and the parameter names`,
  `duplicate output column "<h>": <output> at t<time> appears more than once`,
  `line <n>: expected <M> columns, found <k>`.

### File formats and guards (`upload.ts`)

- Accepted: CSV and Parquet (`.csv,.parquet,.pq`); detection by extension then the
  `PAR1` magic. Unsafe formats (`.pkl`, `.npy`, `.xlsx`, `.h5`, …) are rejected with
  per-format messages. Guards: 512 MB file size, 5M cells.

### Design CSV download

- Header `run_id,<param names>`; `run_id` is the row index `0..N-1`. This is the
  join key users reference in design-route Y uploads.

## The "expected schema" UI

`SchemaGuide.tsx` renders small CSV sample tables under an "expected schema" box in
section 03. It has three states keyed off `xSource`:

- `route="none"` (no X loaded) — neutral message: "Load X first — the file format
  depends on your X source."
- `route="design"` — X design CSV (`run_id,<names>`), Y run_id-aligned (shuffled
  `run_id` to show "any row order"), and time-resolved example.
- `route="uploaded"` — X header exactly the parameter names, Y positional in row
  order, time-resolved example.

The examples derive headers live from `problem.names` and `problem.outputNames` so
they always match the user's problem. Sample tables use the shadcn `Table` primitives
with `font-mono text-xs` + `tabular-nums` cells and an `sr-only` caption (matches the
table style in `DataPanel.tsx`). Keep this UI, the engine parsers, and the tests in
sync — they pin the same contract.

## Method availability

`AnalyzePanel.availability()` pairs each method with its requirement:

- sobol/morris/kucherenko/efast — require their own design (`xSource.kind === "design"`).
- pce/shapley/borgonovo/pawn — fit any input–output data (`xSource.kind === "uploaded"`
  or a design). PCE-family methods reject truncated-gaussian inputs in the browser
  ("Browser PCE does not yet support truncated Gaussian inputs.").

## Tests that pin the contract

- `web/src/site/engine.spec.ts` — `buildDesignCsv`, `classifyYColumn`/`_t{time}`,
  `parseYColumns` classification, `alignColumnsByRunId` permutation checks,
  `parseXGiven`/given-data path.
- `web/src/site/upload.spec.ts` — extension/magic sniffing, unsafe-format rejection,
  Parquet fixture, cell-count guard.
- `web/src/site/AnalyzePanel.spec.ts` — method `availability()` and a PCE run.
- Method math ports under `web/src/jaxgsa/` have their own goldens in `goldens/v0.9.1/`.
- After UI wiring changes, smoke the stage flow end to end: dev server +
  Playwright — hero "Run the 2-minute demo" → analyze stage → "Run selected" →
  results stage.

## Keep in mind

- Run targeted tests after a change (`engine.spec.ts`, `upload.spec.ts`, the method's
  own spec) rather than the full suite — the full `web` suite is fast, but repo-wide
  `pytest` is not.
- `npm run lint` (oxlint), `npx tsc --noEmit`, and `npm run build` in `web/` are the
  verification steps for UI changes.
- The workbench theme follows the system `prefers-color-scheme` with a
  `localStorage` override ("jaxgsa-theme") and a light fallback.