---
name: jaxgsa-web
description: Use when working on the jaxgsa studio (the browser app) under web/, especially the three-step problem → data → analysis flow, the X/Y CSV/Parquet upload contracts, run_id alignment, the expected-schema UI, and the site engine. Covers where each piece lives, what file formats and column layouts the uploaders accept, the exact error strings users can hit, and the tests that pin the contract.
---

# jaxgsa-web

jaxgsa studio, the WASM/React browser app at `web/` (live at https://danielepessina.github.io/jaxgsa/studio/; `/workbench/` is only a redirect page from `web/redirect/workbench.html`), runs ported sensitivity analyses locally in the
browser (float64, no data leaves the page). The user flow is four stages shown
one at a time by the stepper: **01 Problem → 02 Data → 03 Analyze → 04 Results**
(`App.tsx` stage state + `StageNav`; data route readiness via `DataPanel.dataRouteStatus`).
All four stage panels stay mounted; inactive ones use the `hidden` attribute. Never
make a panel's state depend on mounting: opening or collapsing a stage must not
change data. Only a change of the validated problem key clears designs, Y and results.

```ts
web/src/App.tsx                 // stage state machine: stage 0..3, problem, designs, givenX, yData, results
web/src/site/StageNav.tsx       // stepper rail (done/locked states, locked = upstream data missing)
web/src/site/PanelHeading.tsx   // shared panel heading (kicker + title + sub)
web/src/site/ProblemPanel.tsx   // controlled table: rows/outputNames/demoId live in App
web/src/site/problemRows.ts     // row model, validateProblem, problemKey, demo matching, clampStage
web/src/site/session.ts         // localStorage autosave (keys prefixed "jaxgsa-studio:")
web/src/site/designForm.ts      // design form model + validation (power-of-2 base N, ...);
                                // imports isPowerOfTwo, minNPerCurve, saltelliStep from web/src/jaxgsa/
web/src/site/methods.ts         // MethodKey list + per-method blurbs and input kind
web/src/site/analysisGuards.ts  // PCE order parsing, run guard (drops stale runs), warnings
web/src/site/DataPanel.tsx      // ONE place for data: route cards (new design vs. uploads),
                                // design generation (ex-SamplePanel), X/Y uploads, SchemaGuide
web/src/site/AnalyzePanel.tsx   // method selection + run (no uploads); exports `availability()`
web/src/site/FileUpload.tsx     // drag-drop file input (CSV/Parquet)
web/src/site/upload.ts          // format sniffing, unsafe-format rejection, size guards
web/src/site/engine.ts          // parse/align/build helpers (the contract lives here)
web/src/site/SchemaGuide.tsx    // rendered CSV sample tables under "expected schema"
web/src/site/demos.ts           // loadable benchmarks (demo model evaluate)
web/src/site/ResultsPanel.tsx   // results cards, downloads, session JSON
web/src/jaxgsa/                 // ported method math (borgonovo, efast, kucherenko, morris, pawn,
                                // pce, shapley, sobol, plus sampling.ts)
web/e2e/                        // Playwright tests (see web/e2e/README.md)
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
- One numeric row per model run. A `run_id` column is allowed anywhere (so a downloaded
  design CSV works as X). `parseXGiven` returns it separately as `runIds`
  (whole numbers, unique); the X rows keep the file's row order.
- Errors: `the X CSV has no data rows`,
  `the X CSV header must be exactly: <names> (a run_id column is allowed)`,
  `line <n>: expected <k> columns, found <m>`,
  `line <n>: invalid run_id <v> in X (expected a whole number)`,
  `line <n>: duplicate run_id <v> in X`.

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

- If **both** X and Y carry `run_id`, Y is matched to X by id, in any row order
  (`alignColumnsToXRunIds`). Errors: `line <n>: invalid run_id <v> (expected a whole number)`,
  `line <n>: duplicate run_id <v> in Y`, `line <n>: run_id <v> is in Y but not in X`.
- Otherwise Y is **positional**: one value per X row, in the **same row order** as X.
  If X has `run_id` but Y has none, the rows pair by order.
- Positional row count must equal X's row count, else `X has <N> rows but Y has <M> values`.

### Time-resolved outputs (both routes)

- Suffix `_t<number>` — any finite decimal, incl. signs/scientific notation:
  `y_t0`, `y_t0.5`, `y_t3`, `y_t-1`, `y_t2e1`.
- A bare name (no suffix) is that output at time 0. Within an output channel, time
  coordinates are sorted; channels keep first-seen header order.
- Escape hatch: an output literally named `foo_t2` must be written `foo_t2_t0`.
- Errors: `no output columns found: the Y file needs at least one column besides run_id and the parameter names`,
  `duplicate output column "<h>": <output> at t<time> appears more than once`,
  `line <n>: expected <M> columns, found <k>`.
- Parquet has `the parquet file has no data rows` and `file is too large: <r> rows x <c> columns exceeds the <n> cell limit`.

### File formats and guards (`upload.ts`)

- Accepted: CSV and Parquet (`.csv,.parquet,.pq`); detection by extension then the
  `PAR1` magic. Unsafe formats (`.pkl`, `.npy`, `.xlsx`, `.h5`, …) are rejected with
  per-format messages. Guards: 512 MB file size, 5M cells.

### Number and CSV parsing (`engine.ts` `parseCsv` / `parseDecimal`)

- Delimiter is detected from the header line: comma, semicolon or tab.
- Double-quoted fields and headers work (R `write.csv`), as do `""` escapes, a UTF-8
  BOM, CRLF and trailing blank lines. A decimal comma in a `;` or tab file gives a
  targeted error.
- Values must be finite decimals. `Infinity`, `1e999` → "not a finite number";
  `0x10`, `NaN` → "not a number". Parquet cells follow the same rules (`<where>: <v> is not a number`).
- The cell count is estimated before rows are built, so oversized files fail early.
- `validateParameterName` rejects empty names, leading or trailing spaces, commas, quotes,
  semicolons, tabs, line breaks, `run_id` and `_t<number>` suffixes, because they break
  the design CSV header.

### Autosave (`session.ts`)

- Saved (keys `rows`, `output-names`, `demo-id`, `stage`, `data-method`,
  `data-design-form`, `analyze-methods`, `analyze-order`): problem rows, output names,
  demo id, stage, design method and form, method selection, PCE order. Never saved: uploaded data, designs, results. After a reload the stage
  is clamped to the furthest unlocked one. "Reset session" clears every
  `jaxgsa-studio:` key.

### Design CSV download

- Header `run_id,<param names>`; `run_id` is the row index `0..N-1`. This is the
  join key users reference in design-route Y uploads.

## The "expected schema" UI

`SchemaGuide.tsx` renders small CSV sample tables under an "expected schema" box in
step 02 (Data). It has three states keyed off `xSource`:

- `route="none"` (no data route loaded) — neutral message: "Load a data route first — the X
  and Y file layouts depend on it. ..."
- `route="design"` — X design CSV (`run_id,<names>`), Y run_id-aligned (shuffled
  `run_id` to show "any row order"), and time-resolved example.
- `route="uploaded"` — X header exactly the parameter names, Y positional in row
  order (or matched by `run_id` when both files have it), time-resolved example.

The examples derive headers live from `problem.names` and `problem.outputNames` so
they always match the user's problem. Sample tables use the shadcn `Table` primitives
with `font-mono text-xs` + `tabular-nums` cells and an `sr-only` caption (matches the
table style in `DataPanel.tsx`). Keep this UI, the engine parsers, and the tests in
sync — they pin the same contract.

## Method availability

`AnalyzePanel.availability()` pairs each method with its requirement:

The studio supports eight methods (`MethodKey` in `methods.ts`): sobol, morris,
kucherenko, efast, pce, shapley, borgonovo, pawn.

- sobol/morris/kucherenko/efast — require their own design (`xSource.kind === "design"`
  with the same `method`).
- pce/shapley/borgonovo/pawn — fit any input–output data (`xSource.kind === "uploaded"`
  or a design). `pce` and `shapley` (`ORDER_METHODS`) are unavailable with truncated-gaussian
  inputs: "Needs Uniform or untruncated Gaussian inputs. The browser does not support
  truncated Gaussian inputs for this method yet." PAWN and Borgonovo stay available.

## Tests that pin the contract

- `web/src/site/engine.spec.ts` — `buildDesignCsv`, `classifyYColumn`/`_t{time}`,
  `parseYColumns` classification, `alignColumnsByRunId` permutation checks,
  `parseXGiven`/given-data path.
- `web/src/site/upload.spec.ts` — extension/magic sniffing, unsafe-format rejection,
  Parquet fixture, cell-count guard.
- `web/src/site/AnalyzePanel.spec.ts` — method `availability()` and a PCE run.
- `web/src/site/designForm.spec.ts`, `problemRows.spec.ts`, `analysisGuards.spec.ts` —
  design form, problem model and run guard.
- Method math ports under `web/src/jaxgsa/` have their own specs; goldens are in
  the repo-root `goldens/v1.1.0/` (loaded by `web/src/goldens.ts`).
- `web/e2e/*.spec.ts` — Playwright tests (`npm run test:e2e`).
- After UI wiring changes, smoke the stage flow end to end: dev server +
  Playwright — hero "Run the 2-minute demo" → analyze stage → "Run selected" →
  results stage.

## Keep in mind

- Run targeted tests after a change (`engine.spec.ts`, `upload.spec.ts`, the method's
  own spec) rather than the full suite — the full `web` suite is fast, but repo-wide
  `pytest` is not.
- `npm run lint` (oxlint), `npm run typecheck`, `npm test` and `npm run build` in `web/` are
  the verification steps for UI changes.
- The studio theme follows the system `prefers-color-scheme` with a
  `localStorage` override ("jaxgsa-theme") and a light fallback.

## Deploy and CI

- `.github/workflows/docs.yml` builds with `VITE_BASE=/jaxgsa/studio/` and copies `web/dist` to `dist/studio`. It also writes `web/redirect/workbench.html` to `dist/workbench/index.html`.
- `web/public/` holds static files such as `favicon.svg`. `@jax-js/jax` is pinned to an exact version.
- The `web` job in `.github/workflows/ci.yml` runs lint, typecheck, tests, build and the Playwright e2e tests.
