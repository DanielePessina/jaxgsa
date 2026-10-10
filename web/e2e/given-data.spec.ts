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
  type Studio,
} from "./helpers";

const LOW = -Math.PI;

const SPAN = 2 * Math.PI;

interface Cloud {
  x: number[][];
  y: number[];
}

/** Uniform points on the Ishigami box, with the model evaluated at each. */
function ishigamiCloud(n: number, seed: number): Cloud {
  const random = seededRandom(seed);
  const x = Array.from({ length: n }, () => [0, 1, 2].map(() => LOW + SPAN * random()));

  return { x, y: x.map(ishigami) };
}

async function uploadGivenData(studio: Studio, x: number[][], y: number[]): Promise<void> {
  await studio.loadIshigamiProblem();
  await studio.go("data");
  await studio.page.getByRole("button", { name: "I have completed model runs" }).click();
  await studio.fileInput("Upload inputs (X)").setInputFiles(csvFile("x.csv", toCsv(["x1", "x2", "x3"], x)));
  await expect(studio.fileInput("Upload outputs (Y)")).toBeEnabled();
  await studio.fileInput("Upload outputs (Y)").setInputFiles(
    csvFile("y.csv", toCsv(["y"], y.map((v) => [v]))),
  );
  expect(await studio.settleUpload()).toBe("ready");
  await expect(studio.alerts).toHaveCount(0);
}

test("given data: upload X and Y without run_id, run pce, shapley, borgonovo, pawn", async ({ studio }) => {
  const { x, y } = ishigamiCloud(4000, 11);

  await uploadGivenData(studio, x, y);
  await studio.go("analyze");

  for (const method of ["pce", "shapley", "borgonovo", "pawn"]) {
    const box = studio.page.getByRole("checkbox", { name: new RegExp(`^${method}`) });

    await expect(box).toBeEnabled();
    await box.check();
  }

  await studio.page.getByLabel("PCE polynomial order").fill("8");
  await expect(studio.runButton).toContainText("(4)");
  await studio.run();
  await expect(studio.alerts).toHaveCount(0);

  await studio.expectResultCardsRender(4);
  await expect(studio.resultCards).toHaveText(["pce", "shapley", "borgonovo", "pawn"]);

  for (let i = 0; i < 4; i++) {
    const table = await studio.resultsCsv(i);
    const numeric = table.headers.slice(3);

    expect(table.rows).toHaveLength(3);

    for (const label of numeric) {
      expect(column(table, label).every(Number.isFinite), `${label} is finite`).toBe(true);
    }
  }

  // PCE at order 8 on 4000 Ishigami points recovers the analytic S1.
  const pce = await studio.resultsCsv(0);
  const s1 = column(pce, "S1");

  ISHIGAMI_S1.forEach((expected, i) => {
    expect(Math.abs(s1[i] - expected), `pce S1[x${i + 1}]`).toBeLessThan(0.05);
  });

  studio.expectClean();
});

test("given data: Sobol-only methods stay unavailable on uploaded X", async ({ studio }) => {
  const { x, y } = ishigamiCloud(300, 5);

  await uploadGivenData(studio, x, y);
  await studio.go("analyze");

  for (const method of ["sobol", "morris", "kucherenko", "efast"]) {
    await expect(studio.page.getByRole("checkbox", { name: new RegExp(`^${method}`) })).toBeDisabled();
  }
});

test("given data: PCE order 0 cannot run", async ({ studio }) => {
  const { x, y } = ishigamiCloud(300, 5);

  await uploadGivenData(studio, x, y);
  await studio.go("analyze");
  await studio.page.getByRole("checkbox", { name: /^pce/ }).check();
  await studio.page.getByLabel("PCE polynomial order").fill("0");

  // Either the button is off, or the click ends in an error and no result.
  if (await studio.runButton.isEnabled()) {
    await studio.runButton.click();
    await expect(studio.alerts.first()).toBeVisible();
  }

  await expect(studio.resultCards).toHaveCount(0);
});

test("given data: Y is aligned to X by run_id when both files carry it", async ({ studio }) => {
  const { x, y } = ishigamiCloud(2000, 21);
  const random = seededRandom(3);
  const ids = x.map((_, i) => i);
  const xOrder = shuffled(ids, random);
  const yOrder = shuffled(ids, random);

  await studio.loadIshigamiProblem();
  await studio.go("data");
  await studio.page.getByRole("button", { name: "I have completed model runs" }).click();
  await studio.fileInput("Upload inputs (X)").setInputFiles(
    csvFile("x.csv", toCsv(["run_id", "x1", "x2", "x3"], xOrder.map((i) => [i, ...x[i]]))),
  );
  await expect(studio.fileInput("Upload outputs (Y)")).toBeEnabled();
  await studio.fileInput("Upload outputs (Y)").setInputFiles(
    csvFile("y.csv", toCsv(["run_id", "y"], yOrder.map((i) => [i, y[i]]))),
  );
  expect(await studio.settleUpload()).toBe("ready");

  await studio.go("analyze");
  await studio.page.getByRole("checkbox", { name: /^pce/ }).check();
  await studio.page.getByLabel("PCE polynomial order").fill("8");
  await studio.run();

  // Positional pairing of two different shuffles would destroy the signal.
  const s1 = column(await studio.resultsCsv(0), "S1");

  ISHIGAMI_S1.forEach((expected, i) => {
    expect(Math.abs(s1[i] - expected), `pce S1[x${i + 1}]`).toBeLessThan(0.05);
  });
});
