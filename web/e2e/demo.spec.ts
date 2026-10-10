import { column, expect, ISHIGAMI_S1, test } from "./helpers";

const DEMO_METHODS = ["sobol", "pce", "shapley", "borgonovo", "pawn"];

test("hero demo lands on Results with all five method cards", async ({ studio }) => {
  await studio.demoButton.click();

  await expect(studio.stageButton("results")).toHaveAttribute("aria-current", "step", {
    timeout: 120_000,
  });
  await expect(studio.resultCards).toHaveText(DEMO_METHODS);
  await studio.expectResultCardsRender(DEMO_METHODS.length);
  await expect(studio.alerts).toHaveCount(0);
});

test("hero demo: PCE (order 8) S1 matches the Ishigami analytic values", async ({ studio }) => {
  await studio.demoButton.click();
  await expect(studio.resultCards).toHaveCount(DEMO_METHODS.length, { timeout: 120_000 });

  const pce = await studio.resultsCsv(DEMO_METHODS.indexOf("pce"));
  const s1 = column(pce, "S1");

  expect(pce.rows.map((r) => r.parameter)).toEqual(["x1", "x2", "x3"]);
  expect(Math.abs(s1[1] - 0.442), "PCE S1[x2]").toBeLessThan(0.03);
  ISHIGAMI_S1.forEach((expected, i) => {
    expect(Math.abs(s1[i] - expected), `PCE S1[x${i + 1}]`).toBeLessThan(0.03);
  });
});

test("a clean load has no console errors, page errors or failed requests", async ({ studio, page }) => {
  // The favicon is requested on load, so a missing one shows up here.
  await page.waitForLoadState("networkidle");
  await expect(page.locator("link[rel~=icon]").first()).toHaveAttribute("href", /.+/);
  studio.expectClean();
});

test("the demo run has no console errors or page errors", async ({ studio, page }) => {
  await studio.demoButton.click();
  await expect(studio.resultCards).toHaveCount(DEMO_METHODS.length, { timeout: 120_000 });
  await page.waitForLoadState("networkidle");
  studio.expectClean();
});
