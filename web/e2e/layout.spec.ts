import { expect, test, type Studio } from "./helpers";

test.describe("390px viewport", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  async function expectNoHorizontalScroll(studio: Studio, where: string): Promise<void> {
    const { scrollWidth, clientWidth } = await studio.page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }));

    expect(scrollWidth, `page is wider than the screen on ${where}`).toBeLessThanOrEqual(clientWidth);
  }

  test("no horizontal page scroll on any step", async ({ studio }) => {
    await expectNoHorizontalScroll(studio, "load");

    await studio.demoButton.click();
    await expect(studio.resultCards).toHaveCount(5, { timeout: 120_000 });
    await expectNoHorizontalScroll(studio, "results");

    for (const stage of ["analyze", "data", "problem"] as const) {
      await studio.go(stage);
      await expectNoHorizontalScroll(studio, stage);
    }
  });

  test("the step rail shows all four steps inside the screen", async ({ studio }) => {
    for (const stage of ["problem", "data", "analyze", "results"] as const) {
      const button = studio.stageButton(stage);

      await expect(button).toBeVisible();

      const box = await button.boundingBox();

      expect(box, `${stage} step has a box`).not.toBeNull();
      expect(box?.x ?? -1).toBeGreaterThanOrEqual(0);
      expect((box?.x ?? 0) + (box?.width ?? 0)).toBeLessThanOrEqual(390);
    }
  });
});

test.describe("keyboard", () => {
  /** Tab until the focused element's text or label matches. */
  async function tabTo(studio: Studio, pattern: RegExp, key = "Tab"): Promise<void> {
    for (let i = 0; i < 120; i++) {
      await studio.page.keyboard.press(key);

      const name = await studio.page.evaluate(() => {
        const el = document.activeElement;

        return `${el?.getAttribute("aria-label") ?? ""} ${el?.textContent ?? ""}`.trim();
      });

      if (pattern.test(name)) return;
    }

    throw new Error(`${pattern} was not reachable with ${key}`);
  }

  test("demo, run and download are reachable with Tab and Enter", async ({ studio, page }) => {
    await tabTo(studio, /Run the demo/);
    await page.keyboard.press("Enter");
    await expect(studio.resultCards).toHaveCount(5, { timeout: 120_000 });
    await expect(studio.stageButton("results")).toHaveAttribute("aria-current", "step");

    // Download a results CSV from the keyboard.
    await tabTo(studio, /^results csv$/);

    const [download] = await Promise.all([page.waitForEvent("download"), page.keyboard.press("Enter")]);

    expect(download.suggestedFilename()).toMatch(/results\.csv$/);

    // Go back to Analyze and run once more, again from the keyboard.
    await tabTo(studio, /Analyze$/, "Shift+Tab");
    await page.keyboard.press("Enter");
    await tabTo(studio, /^Run selected/);
    await page.keyboard.press("Enter");
    await expect.poll(() => studio.resultCards.count(), { timeout: 120_000 }).toBeGreaterThan(5);
  });

  test("the Y upload opens a file chooser from the keyboard", async ({ studio, page }) => {
    await studio.generateDesign("sobol", { "Base N": "16" });
    await tabTo(studio, /Upload outputs \(Y\)/);

    const chooser = page.waitForEvent("filechooser");

    await page.keyboard.press("Enter");
    expect(await chooser).toBeTruthy();
  });
});
