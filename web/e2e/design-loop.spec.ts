import {
  column,
  csvFile,
  expect,
  ISHIGAMI_S1,
  ishigami,
  seededRandom,
  shuffled,
  test,
  toCsv,
  type DesignMethod,
} from "./helpers";

interface Case {
  method: DesignMethod;
  fields: Record<string, string>;
}

const CASES: Case[] = [
  { method: "sobol", fields: { "Base N": "4096" } },
  { method: "morris", fields: { Trajectories: "50" } },
  { method: "kucherenko", fields: { Samples: "4096" } },
  { method: "efast", fields: { "Points per curve": "1025" } },
];

for (const { method, fields } of CASES) {
  test(`design loop: ${method} generate, download, evaluate, upload shuffled Y, run, export`, async ({
    studio,
  }) => {
    await studio.loadIshigamiProblem();
    await studio.generateDesign(method, fields);

    const design = await studio.downloadDesign();

    expect(design.headers).toEqual(["run_id", "x1", "x2", "x3"]);
    expect(design.rows.map((r) => Number(r[0]))).toEqual(design.rows.map((_, i) => i));

    // Evaluate the model here, then shuffle the rows: run_id must realign them.
    const evaluated = design.rows.map((r) => [r[0], ishigami(r.slice(1, 4).map(Number))]);
    const yText = toCsv(["run_id", "y"], shuffled(evaluated, seededRandom(7)));

    await studio.uploadY(csvFile(`${method}-y.csv`, yText));
    expect(await studio.settleUpload()).toBe("ready");
    await expect(studio.alerts).toHaveCount(0);

    await studio.go("analyze");
    await expect(studio.runButton).toContainText("(1)");
    await studio.run();
    await expect(studio.alerts).toHaveCount(0);

    await studio.expectResultCardsRender(1);
    await expect(studio.resultCards.first()).toHaveText(new RegExp(method, "i"));

    const table = await studio.resultsCsv(0);

    expect(table.rows.map((r) => r.parameter)).toEqual(["x1", "x2", "x3"]);

    const index = method === "morris" ? "mu*" : "S1";
    const values = column(table, index);

    expect(values.every(Number.isFinite)).toBe(true);

    if (method !== "morris") {
      // x2 carries the most first-order variance, x3 the least.
      expect(values[1]).toBeGreaterThan(values[0]);
      expect(values[0]).toBeGreaterThan(values[2]);
    }

    if (method === "sobol") {
      ISHIGAMI_S1.forEach((expected, i) => {
        expect(Math.abs(values[i] - expected), `S1[x${i + 1}]`).toBeLessThan(0.03);
      });
    }

    // The page itself must stay quiet through the whole loop.
    studio.expectClean();
  });
}
