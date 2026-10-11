import { expect, test } from "./helpers";

test("revisit: opening step 01 after results keeps designs, results and the problem", async ({ studio }) => {
  await studio.demoButton.click();
  await expect(studio.resultCards).toHaveCount(5, { timeout: 120_000 });

  const before = await studio.problemSnapshot();

  await studio.go("problem");
  expect(await studio.problemSnapshot()).toEqual(before);
  expect(before.names).toEqual(["x1", "x2", "x3"]);

  // The later steps are still open and still hold the data.
  for (const stage of ["data", "analyze", "results"] as const) {
    await expect(studio.stageButton(stage)).toBeEnabled();
  }

  await studio.go("data");
  await expect(studio.page.getByText(/uploaded Y|demo model/).first()).toBeVisible();
  await expect(studio.nextToMethods).toBeEnabled();

  await studio.go("results");
  await expect(studio.resultCards).toHaveCount(5);
  expect(await studio.problemSnapshot()).toEqual(before);
});

test("revisit: a custom problem survives leaving step 01 and coming back", async ({ studio }) => {
  await studio.problemNameInputs.first().fill("alpha");
  await studio.problemHighInputs.first().fill("5");

  await studio.go("data");
  await studio.page.getByRole("button", { name: "Start a new experiment" }).click();
  await studio.page.getByRole("button", { name: /^(Generate|Regenerate) design$/ }).click();
  await expect(studio.page.getByRole("button", { name: "Download design CSV" })).toBeVisible();

  await studio.go("problem");

  const after = await studio.problemSnapshot();

  expect(after.names[0]).toBe("alpha");
  expect(after.highs[0]).toBe("5");
  expect(after.names.slice(1)).toEqual(["x2", "x3"]);

  // The generated design is still there too.
  await studio.go("data");
  await expect(studio.page.getByRole("button", { name: "Download design CSV" })).toBeVisible();
  const design = await studio.downloadDesign();

  expect(design.headers).toEqual(["run_id", "alpha", "x2", "x3"]);
});

test("output names update the expected-schema guide", async ({ studio, page }) => {
  await studio.go("problem");
  await page.locator("#output-names").fill("temp, pressure");
  await studio.go("data");

  // The guide shows the first output name in its examples.
  const data = page.locator("#stage-data");

  await expect(data).toContainText("temp");

  await studio.go("problem");
  await page.locator("#output-names").fill("flow");
  await studio.go("data");
  await expect(data).toContainText("flow");
  await expect(data).not.toContainText("temp");
});

test("autosave: a reload restores the problem table", async ({ studio, page }) => {
  await studio.problemNameInputs.first().fill("alpha");
  await studio.problemHighInputs.nth(1).fill("7");
  await page.locator("#output-names").fill("temp");

  const saved = await studio.problemSnapshot();

  await page.reload();
  await expect(studio.demoButton).toBeEnabled({ timeout: 90_000 });

  expect(await studio.problemSnapshot()).toEqual(saved);
  expect(saved.names[0]).toBe("alpha");
});

test("autosave: Reset session clears the saved problem", async ({ studio, page }) => {
  await studio.problemNameInputs.first().fill("alpha");
  await studio.problemHighInputs.nth(1).fill("7");

  page.once("dialog", (dialog) => void dialog.accept());
  await page.getByRole("button", { name: "Reset session" }).click();

  await expect(studio.problemNameInputs.first()).toHaveValue("x1");
  await expect(studio.problemHighInputs.nth(1)).toHaveValue("1");

  await page.reload();
  await expect(studio.demoButton).toBeEnabled({ timeout: 90_000 });
  await expect(studio.problemNameInputs.first()).toHaveValue("x1");
  await expect(studio.problemHighInputs.nth(1)).toHaveValue("1");
});

test("autosave: Reset session after results returns to an empty session", async ({ studio, page }) => {
  await studio.demoButton.click();
  await expect(studio.resultCards).toHaveCount(5, { timeout: 120_000 });

  page.once("dialog", (dialog) => void dialog.accept());
  await page.getByRole("button", { name: "Reset session" }).click();

  await expect(studio.stageButton("problem")).toHaveAttribute("aria-current", "step");
  await expect(studio.stageButton("results")).toBeDisabled();
  await expect(studio.resultCards).toHaveCount(0);
});
