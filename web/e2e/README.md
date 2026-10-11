# jaxgsa studio end-to-end tests

Playwright tests that drive the real app in headless Chromium.

## Run

```sh
cd web
npm ci
npm run test:e2e
```

The first run builds the app into `e2e/.build` (served at `/`) and starts `vite preview`.
The tests see the production bundle.
Locally, a preview server already on port 4173 is reused. Stop it to force a fresh build.

| Variable | Effect |
| --- | --- |
| `E2E_BASE_URL` | Test a server that already runs (for example `npm run dev`). No build. |
| `E2E_CHANNEL` | Browser channel. Default: `chrome` (installed Google Chrome) locally, `chromium` in CI. |
| `CI` | Two workers, one retry, HTML report. |

Without Google Chrome, install the Playwright browser once:

```sh
npx playwright install --with-deps chromium
E2E_CHANNEL=chromium npm run test:e2e
```

Run one file: `npm run test:e2e -- design-loop`. Debug: add `--headed` or `--debug`.

## Files

| File | Covers |
| --- | --- |
| `design-loop.spec.ts` | Sobol, Morris, Kucherenko, eFAST: generate, download, upload shuffled Y, run, export |
| `given-data.spec.ts` | Own X and Y: PCE, Shapley, Borgonovo, PAWN; PCE order 0 |
| `demo.spec.ts` | Hero demo, PCE accuracy, zero console errors |
| `regressions.spec.ts` | Revisiting steps, output names, autosave, Reset session |
| `uploads.spec.ts` | Malformed and accepted Y files |
| `layout.spec.ts` | 390px layout, keyboard-only use |
| `helpers.ts` | `Studio` page object, Ishigami, CSV helpers |

## Rules

- Prefer role and text locators over CSS. The analysis panels change often.
- Do not weaken an assertion to make a test pass. A failing test is a bug report.
- Generate files in memory (`setInputFiles` with a buffer). Do not write fixtures to disk.
