import { readFile } from "node:fs/promises";
import { expect, test as base, type Locator, type Page } from "@playwright/test";

export type Stage = "problem" | "data" | "analyze" | "results";

export type DesignMethod = "sobol" | "morris" | "kucherenko" | "efast";

/** The Ishigami function with the studio's constants (a = 7, b = 0.1). */
export function ishigami(x: readonly number[]): number {
  return Math.sin(x[0]) + 7 * Math.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * Math.sin(x[0]);
}

/** Analytic first-order indices of Ishigami (a = 7, b = 0.1). */
export const ISHIGAMI_S1 = [0.3139, 0.4424, 0.0];

/** Small seeded generator, so a failing run can be replayed. */
export function seededRandom(seed: number): () => number {
  let a = seed;

  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffled<T>(items: readonly T[], random: () => number): T[] {
  const out = [...items];

  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }

  return out;
}

export interface Csv {
  headers: string[];
  rows: string[][];
}

export function parseCsv(text: string): Csv {
  const lines = text.trim().split(/\r?\n/);

  return {
    headers: lines[0].split(","),
    rows: lines.slice(1).map((l) => l.split(",")),
  };
}

export function toCsv(headers: readonly string[], rows: readonly (readonly (string | number)[])[]): string {
  return [headers.join(","), ...rows.map((r) => r.join(","))].join("\n") + "\n";
}

export interface UploadFile {
  name: string;
  mimeType: string;
  buffer: Buffer;
}

export function csvFile(name: string, text: string): UploadFile {
  return { name, mimeType: "text/csv", buffer: Buffer.from(text) };
}

/** A parsed "results csv" download: one row per parameter and output slice. */
export interface ResultsTable {
  headers: string[];
  rows: Record<string, string>[];
}

export function parseResultsCsv(text: string): ResultsTable {
  const { headers, rows } = parseCsv(text);

  return {
    headers,
    rows: rows.map((r) => Object.fromEntries(headers.map((h, i) => [h, r[i]]))),
  };
}

/** Index values of one column, in parameter order. */
export function column(table: ResultsTable, label: string): number[] {
  expect(table.headers, `results csv has a ${label} column`).toContain(label);

  return table.rows.map((r) => Number(r[label]));
}

/**
 * Page object for the studio. It also records console errors, page errors and
 * failed requests, so a test can assert that a flow stayed clean.
 */
export class Studio {
  readonly problems: string[] = [];

  constructor(readonly page: Page) {
    page.on("console", (m) => {
      if (m.type() === "error") this.problems.push(`[console.error] ${m.text()}`);
    });
    page.on("pageerror", (e) => this.problems.push(`[pageerror] ${e.message}`));
    page.on("requestfailed", (r) => {
      this.problems.push(`[requestfailed] ${r.url()} ${r.failure()?.errorText ?? ""}`);
    });
    page.on("response", (r) => {
      if (r.status() >= 400) this.problems.push(`[http ${r.status()}] ${r.url()}`);
    });
  }

  /** Open the app and wait until the engine is ready (the demo button is on). */
  async open(): Promise<void> {
    await this.page.goto("/");
    await expect(this.demoButton).toBeEnabled({ timeout: 90_000 });
  }

  get demoButton(): Locator {
    return this.page.getByRole("button", { name: "Run the 2-minute demo" });
  }

  get alerts(): Locator {
    return this.page.getByRole("alert");
  }

  get runButton(): Locator {
    return this.page.getByRole("button", { name: /^Run selected/ });
  }

  get nextToMethods(): Locator {
    return this.page.getByRole("button", { name: /^Choose the methods/ });
  }

  get resultCards(): Locator {
    return this.page.locator("#stage-results h3");
  }

  get resultsCsvButtons(): Locator {
    return this.page.locator("#stage-results").getByRole("button", { name: "results csv" });
  }

  stageButton(stage: Stage): Locator {
    const label = { problem: "Problem", data: "Data", analyze: "Analyze", results: "Results" }[stage];

    return this.page
      .getByRole("navigation", { name: "workflow steps" })
      .getByRole("button", { name: new RegExp(`${label}$`) });
  }

  async go(stage: Stage): Promise<void> {
    await this.stageButton(stage).click();
    await expect(this.stageButton(stage)).toHaveAttribute("aria-current", "step");
  }

  /** Fail with every console error, page error and failed request seen so far. */
  expectClean(): void {
    expect(this.problems, "console errors, page errors and failed requests").toEqual([]);
  }

  /** Load the Ishigami benchmark through the Problem step ("Start Sobol demo"). */
  async loadIshigamiProblem(): Promise<void> {
    await this.go("problem");
    await this.page
      .locator("#stage-problem select")
      .filter({ has: this.page.locator("option[value=ishigami]") })
      .selectOption("ishigami");
    await this.page.getByRole("button", { name: "Start Sobol demo" }).click();
    await expect(this.problemNameInputs).toHaveCount(3);
    await expect(this.problemHighInputs.first()).toHaveValue(/^3\.14/);
  }

  get problemNameInputs(): Locator {
    return this.page.locator("#stage-problem input[placeholder=x1]");
  }

  get problemHighInputs(): Locator {
    return this.page.locator("#stage-problem input[placeholder=high]");
  }

  async problemSnapshot(): Promise<{ names: string[]; highs: string[]; outputs: string }> {
    return {
      names: await this.problemNameInputs.evaluateAll((els) => els.map((e) => (e instanceof HTMLInputElement ? e.value : ""))),
      highs: await this.problemHighInputs.evaluateAll((els) => els.map((e) => (e instanceof HTMLInputElement ? e.value : ""))),
      outputs: await this.page.locator("#output-names").inputValue(),
    };
  }

  /** Pick a design method on the Data step, fill its fields and generate. */
  async generateDesign(method: DesignMethod, fields: Record<string, string> = {}): Promise<void> {
    await this.go("data");
    await this.page.getByRole("button", { name: "Start a new experiment" }).click();
    await this.page.getByRole("button", { name: new RegExp(`^${method}`, "i") }).click();

    for (const [label, value] of Object.entries(fields)) {
      await this.page.getByLabel(label, { exact: true }).fill(value);
    }

    await this.page.getByRole("button", { name: /^(Generate|Regenerate) design$/ }).click();
    await expect(this.page.getByRole("button", { name: "Download design CSV" })).toBeVisible();

    const use = this.page.getByRole("button", { name: "Use this design" });

    if ((await use.count()) > 0) await use.click();
    await expect(this.page.getByText("in use", { exact: true })).toBeVisible();
  }

  /** Click a button that starts a download and return the file text. */
  async download(click: () => Promise<void>): Promise<{ name: string; text: string }> {
    const [dl] = await Promise.all([this.page.waitForEvent("download"), click()]);
    const path = await dl.path();

    return { name: dl.suggestedFilename(), text: await readFile(path, "utf8") };
  }

  async downloadDesign(): Promise<Csv> {
    const file = await this.download(() =>
      this.page.getByRole("button", { name: "Download design CSV" }).click(),
    );

    return parseCsv(file.text);
  }

  /** The Y file input of the Data step (design route) or the n-th input (given route). */
  fileInput(label: "Upload inputs (X)" | "Upload outputs (Y)"): Locator {
    return this.page.getByLabel(label, { exact: true });
  }

  async uploadY(file: UploadFile): Promise<void> {
    await this.fileInput("Upload outputs (Y)").setInputFiles(file);
  }

  /** Wait until an upload either failed (an alert shows) or the data is ready. */
  async settleUpload(): Promise<"error" | "ready"> {
    await expect(async () => {
      const failed = (await this.alerts.count()) > 0;
      const ready = await this.nextToMethods.isEnabled();
      expect(failed || ready).toBe(true);
    }).toPass({ timeout: 15_000 });

    return (await this.alerts.count()) > 0 ? "error" : "ready";
  }

  /** Run the selected methods and wait for the run to end. */
  async run(): Promise<void> {
    await this.go("analyze");
    await expect(this.runButton).toBeEnabled();
    await this.runButton.click();
    await expect(this.page.getByRole("button", { name: /^Running|^Computing/ })).toHaveCount(0, {
      timeout: 150_000,
    });
  }

  async resultsCsv(index: number): Promise<ResultsTable> {
    await this.go("results");
    const file = await this.download(() => this.resultsCsvButtons.nth(index).click());

    return parseResultsCsv(file.text);
  }

  /** Check that every result card shows a table and a chart. */
  async expectResultCardsRender(count: number): Promise<void> {
    await this.go("results");
    await expect(this.resultCards).toHaveCount(count);
    await expect(this.page.locator("#stage-results table")).toHaveCount(count);
    await expect(this.page.locator("#stage-results svg[role=img]")).toHaveCount(count);
  }
}

export const test = base.extend<{ studio: Studio }>({
  studio: async ({ page }, use) => {
    const studio = new Studio(page);
    await studio.open();
    await use(studio);
  },
});

export { expect };
