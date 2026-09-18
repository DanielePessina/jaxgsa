# PLAN-WEB-UI: jaxgsa web workbench — schema, formats, visualization, UX

Local planning doc (untracked, like PLAN-WASM.md). Source of truth for the
browser app remains the Python package (`src/jaxgsa`) and the wasm port plan
(`PLAN-WASM.md`). This doc is the executable spec for the web UI work.

Status baseline (verified 2026-09-08): `web/` builds, `npm test` = 105 pass,
`npm run typecheck` clean. Last commit: `93ab147` ("one-page GSA workbench").

## Goal

Turn the current one-page stacked workflow into a polished two-column GSA
workbench that supports the full exchange contract: users sample a design in
the browser, download it, evaluate it with their own model, upload outputs
that may have **multiple outputs** and/or **time points**, run analyses, and
read back indices with publication-quality charts they can save.

Everything stays client-side (GitHub Pages static). No server compute.

## Decisions (closed with the user, do not re-litigate)

| # | Topic | Decision |
|---|---|---|
| D1 | Y schema | **Wide format + `Problem.outputNames`.** `_t{t}` suffix is the reserved timepoint marker. X columns allowed & ignored in Y files. Default scalar output name `y`. |
| D2 | File formats | **CSV + Parquet read, CSV/JSON write.** Upload accepts `.csv` and `.parquet` (via `hyparquet`, pure-JS). Downloads remain CSV + session JSON. `.pkl/.joblib/.npy/.npz/.pt` rejected with a clear "unsafe/unsupported" message (pickle = arbitrary code execution). |
| D3 | Plotting | **AG Charts** (`ag-charts-community` + `ag-charts-react`) as the working default, isolated behind an `IndicesChart` component so **Reaviz** is a one-file swap (user still deciding; this is the "decide later" slot). |
| D4 | Typography | **Geist Sans + Geist Mono**, self-hosted via `@fontsource`. |
| D4b | Aesthetic (decided) | **Editorial-scientific**: larger type, generous whitespace, soft borders, restrained accent, less instrument-chrome (reduce the mono-uppercase micro-label pattern). |
| D4c | Animation (decided) | **Full Motion pass**: `motion` (framer-motion v12), copy-paste animated primitives (step rail, collapsible cards, slice tabs, results-dock transitions). |
| D5 | Layout | **Two-column workbench.** Left = Problem → Sample → Analyze (collapsible cards + step-status rail). Right = sticky Results dock, live-updating. Stacks to a single column below `xl`. |
| D6 | Analysis scope | **Schema + full multi-output/time analysis.** Ports stay per-slice scalar; engine loops over slices. |

Deferred (already documented in PLAN-WASM.md): hdmr, S2 estimator, correlation
sampling, categorical marginals, WebGPU float32 acceleration.

## Y exchange schema contract (D1) — THE SPEC

### Problem extension

`ProblemSpec` (web/src/jaxgsa/sampling.ts) gains:

```ts
outputNames?: string[];  // optional labels; default output name = "y"
```

Mirrors Python `Problem.output_names`. Used for labeling + validating Y
columns, never for the layout math.

### Y file layout

Y is a wide CSV/Parquet table. One row per run. Columns are classified in
this exact order:

1. `run_id` → the row key (design workflow). Not a data column.
2. Exact match against problem parameter names → X column, **ignored**
   (so a user may upload the downloaded design CSV with outputs appended).
3. Everything else → an **output column**.

Output column name parsing:

- Match `^(?<base>.+)_t(?<t>[0-9]+)$` → output `base` at time `t`.
- Otherwise → output `<name>` at time `t=0`.
- Reserved rule: an output `base` must not itself end in `_t<digits>`
  (`foo_t2_t0` is the escape hatch for an output literally named `foo_t2`).
  Validated with a clear error.

Examples:

```
run_id,y                        -> output y, T=1
run_id,y_t0,y_t1,y_t2           -> output y, T=3
run_id,temp,pressure            -> outputs {temp, pressure}, T=1 each
run_id,temp_t0,pressure_t0,temp_t1,pressure_t1  -> 2 outputs x 2 times
run_id,x1,x2,y_t0,y_t1          -> X columns x1,x2 ignored, y over T=2
```

Normalized result (the `YData` model):

```ts
interface YColumn { output: string; time: number; values: Float64Array; }
interface YData {
  columns: YColumn[];          // ordered as first-seen in the header
  rowCount: number;            // runs
  label: string;               // human summary for the UI
}
```

Validation rules:

- All data rows have the same column count (existing `parseCsv` behavior).
- ≥ 1 output column after classification.
- No duplicate `(output, time)` pair (e.g. `y_t0` twice) → error naming both.
- Time indices may be sparse (`y_t0, y_t3`) — slices are independent.
- Design workflow: `run_id` required, must be a permutation of `0..rowCount-1`
  (existing `alignYByRunId` semantics, generalized to every column).
- Given-data workflow: positional, N rows must equal X's N; `run_id` if present
  is ignored (X upload has no run_id today).

### Results model

`AnalysisResult` becomes slice-indexed (breaking change, update all users):

```ts
interface ResultSlice {
  output: string;
  time: number;                // 0 when no _t suffix
  columns: ResultColumn[];     // {key,label,values} as today
}
interface AnalysisResult {
  method: string;
  parameters: string[];
  slices: ResultSlice[];
  notes: string[];
  settings: Record<string, unknown>;  // seed, base_n, order, ... for repro
}
```

Scalar Y ⇒ exactly one slice `{output:"y", time:0}` (UI hides the time axis).

### Engine changes (web/src/site/engine.ts)

- `analyzeGenerated` / `analyzeCloud`: loop over `YData.columns`, call the
  existing scalar port per slice, build `ResultSlice[]`. **Ports unchanged.**
- `resultToCsv`: combined CSV (rows = slices, cols = index columns) plus a
  per-slice CSV; both keep the `parameter` row format.
- `buildSessionJson`: bump `version` → 2, store `outputNames`, `slices`,
  per-slice `settings`, and Y as `YData` (columns with output/time labels).

### Demo path

`loadDemoCloud` / `demo.evaluate` produce scalar Y ⇒ one slice. No port change.

## File formats (D2)

- Add `hyparquet` (pure-JS parquet reader, no wasm, browser-safe).
- New `FileUpload` (replaces `CsvUpload`): accepts `.csv` / `.parquet` /
  `.gz`-wrapped csv optionally; detects by extension + content sniff
  (parquet magic `PAR1`); rejects the unsafe/unsupported list with a
  per-format message (pickle-family: "cannot load safely — pickle executes
  arbitrary code; use CSV or Parquet").
- Unified `parseUpload(textOrBytes, hint)` → `ParsedTable`
  (headers + rows), so X and Y uploads share one path.
- Size guard: cap ~5e6 cells (soft), clear error instead of a frozen tab.
- Downloads unchanged: design CSV, results CSV (now slice-aware), session
  JSON (v2).

## Visualization (D3/D4)

- `IndicesChart` (web/src/site/IndicesChart.tsx) — grouped vertical bars:
  x = parameters (category), one series per index column
  (S1/ST or mu/mu*/sigma or Sh/S1/ST).
- Slice selector (tabs/select) when >1 `(output,time)` slice.
- AG Charts theme: `params.fontFamily` = Geist Mono; palette from the dark
  theme chart colors; gridline **major + minor** via `gridLine.style`
  array; minor ticks tuned with `interval.step` (may need per-axis config —
  iterate with user).
- Save: `AgChartInstance.download()` → PNG + a per-chart data CSV
  (chart data as a pure exported `shapeChartData(result, slice)` fn, tested).
- Demo reference lines (S1/ST analytical) drawn as annotations when the
  problem matches a demo.
- Results cards: header (method + settings chips + slice selector) → chart →
  compact table (kept from `ResultsTable`) → notes → download row
  (chart PNG, chart data CSV, results CSV).

## Workbench layout (D5)

- `App.tsx`: `xl`+ = two columns. Left: collapsible Problem/Sample/Analyze
  cards with a compact step-status rail (✓ done / … active / needs input),
  clickable to scroll. Right: sticky (top, self-scrolling) Results dock that
  updates live. Below `xl`: current stacked flow.
- AnalyzePanel UX: X-source picker surfaces the active design's settings
  (seed/base_n/… from `GeneratedDesign.summary`); method availability shows
  explicit badges (RUNNABLE / NEEDS <method> design / reason text);
  `order` setting persists into `results.settings`.
- Header/footer/alert cleanup; fonts swapped everywhere.

## Task list (execution order, subagent-friendly)

**Phase 1 — schema + engine (foundation, no UI)**
1. `outputNames` in `ProblemSpec`; ProblemPanel output-names input.
2. `engine.ts`: `YColumn`/`YData`, `parseYColumns` (spec above),
   `alignYByRunId` → slice-aware, `AnalysisResult.slices`,
   slice-loop analysis, `resultToCsv` slice-aware, session JSON v2.
3. Tests: engine.spec.ts (column classifier incl. reserved-rule errors,
   multi-output alignment, slice analysis on a 2-output Ishigami-style
   stub, session v2).

**Phase 2 — file formats**
4. `hyparquet` dep, `parseUpload` + sniff + reject list + size guard.
5. `FileUpload` component replacing `CsvUpload` (both call sites).
6. Tests (sniff/reject/parquet fixture if practical).

**Phase 3 — typography + theming**
7. `@fontsource` geist-sans + geist-mono; `index.css` theme vars
   (`--font-sans`, `--font-mono`); sweep hardcoded stacks.

**Phase 4 — results visualization**
8. `ag-charts-react` dep; `shapeChartData` (pure) + `IndicesChart`
   (grouped bars, slice selector, minor gridlines/ticks, Geist theme,
   PNG + data-CSV download).
9. Rewrite `ResultsPanel`/`ResultsTable` into result cards.
10. Tests: `shapeChartData` + session round-trip.

**Phase 5 — workbench layout + analyze UX**
11. `App.tsx` two-column + step-status rail + collapsible cards.
12. `AnalyzePanel`: design settings surface, availability badges.
13. Responsive + polish pass.

**Phase 6 — UI iteration loop (user drives)**
14. Multiple feedback rounds: chart styling, spacing, copy, color,
    minor-tick tuning, layout refinements. Each round = small subagent
    diffs reviewed by the user.

**Phase 7 — validation + docs**
15. Full green: `npm run typecheck`, `npm test`, `npm run build`.
16. Document the Y exchange format (site help text + a short
    `docs/guide/web-format.md` note in the VitePress site if desired).

## Verification

```bash
cd web
npm run typecheck   # must be clean
npm test            # must stay green (105 + new)
npm run build       # must succeed
npm run dev         # manual UI iteration
```

## Risks / notes

- AG Charts minor-gridline API may need axis `interval` + theme overrides to
  look "matplotlib-like"; the exact look is a Phase 6 iteration target, not a
  Phase 4 gate.
- `AnalysisResult.slices` is a breaking model change — update ResultsPanel,
  ResultsTable, AnalyzePanel, App, engine.spec.ts, buildSessionJson in the
  same PR.
- Parquet upload path depends on `hyparquet` working on the target browsers;
  verify with a real file early in Phase 2.
- Keep `web/` separate from `docs/` (PLAN-WASM.md rule).

## Open slots (user decides during execution)

- Chart lib: AG Charts (default, pinned) vs Reaviz — swap in `IndicesChart`.
- Exact chart style (gridlines, ticks, palette, reference annotations) — editorial-scientific: muted gridlines, restrained palette, Geist.
- Whether parquet download is ever wanted (currently read-only).
- Editorial restyle of the mono-uppercase micro-label pattern (Phase 5/6).