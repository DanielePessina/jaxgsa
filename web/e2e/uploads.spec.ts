import { csvFile, expect, ishigami, test, toCsv, type Csv, type Studio, type UploadFile } from "./helpers";

interface Fixture {
  ids: string[];
  y: number[];
}

/** A small Sobol' design on the Ishigami box with its model values. */
async function smallDesign(studio: Studio): Promise<Fixture> {
  await studio.loadIshigamiProblem();
  await studio.generateDesign("sobol", { "Base N": "16" });

  const design: Csv = await studio.downloadDesign();

  return {
    ids: design.rows.map((r) => r[0]),
    y: design.rows.map((r) => ishigami(r.slice(1, 4).map(Number))),
  };
}

/** A valid `run_id,y` file with one cell replaced. */
function yCsv({ ids, y }: Fixture, cell?: { row: number; value: string | number }): string {
  return toCsv(
    ["run_id", "y"],
    ids.map((id, i) => [id, cell && cell.row === i ? cell.value : y[i]]),
  );
}

interface Rejected {
  name: string;
  file: (f: Fixture) => UploadFile;
}

const REJECTED: Rejected[] = [
  { name: "Infinity", file: (f) => csvFile("inf.csv", yCsv(f, { row: 3, value: "Infinity" })) },
  { name: "1e999", file: (f) => csvFile("big.csv", yCsv(f, { row: 3, value: "1e999" })) },
  { name: "0x10", file: (f) => csvFile("hex.csv", yCsv(f, { row: 3, value: "0x10" })) },
  { name: "NaN", file: (f) => csvFile("nan.csv", yCsv(f, { row: 3, value: "NaN" })) },
  {
    name: "wrong row count",
    file: (f) => csvFile("short.csv", yCsv({ ids: f.ids.slice(0, -5), y: f.y.slice(0, -5) })),
  },
  { name: "empty file", file: () => csvFile("empty.csv", "") },
  {
    name: ".pkl file",
    file: () => ({ name: "y.pkl", mimeType: "application/octet-stream", buffer: Buffer.from("\x80\x04K\x01.", "latin1") }),
  },
  {
    // Row 1 gets the id of row 0, so one id repeats and one is missing.
    name: "duplicate run_id",
    file: (f) =>
      csvFile("dup.csv", toCsv(["run_id", "y"], f.ids.map((id, i) => [i === 1 ? f.ids[0] : id, f.y[i]]))),
  },
];

for (const { name, file } of REJECTED) {
  test(`malformed upload: ${name} shows an error and no success`, async ({ studio }) => {
    const fixture = await smallDesign(studio);

    await studio.uploadY(file(fixture));

    expect(await studio.settleUpload()).toBe("error");
    await expect(studio.alerts.first()).toBeVisible();
    await expect(studio.nextToMethods).toBeDisabled();
    await expect(studio.page.getByText(/uploaded Y ·/)).toHaveCount(0);
  });
}

test("malformed upload: an error clears after a valid file is uploaded", async ({ studio }) => {
  const fixture = await smallDesign(studio);

  await studio.uploadY(csvFile("nan.csv", yCsv(fixture, { row: 3, value: "NaN" })));
  expect(await studio.settleUpload()).toBe("error");

  await studio.uploadY(csvFile("ok.csv", yCsv(fixture)));
  await expect(studio.nextToMethods).toBeEnabled();
  await expect(studio.alerts).toHaveCount(0);
});

test("accepted upload: a quoted-header CSV", async ({ studio }) => {
  const fixture = await smallDesign(studio);
  const text = yCsv(fixture).replace("run_id,y", '"run_id","y"');

  await studio.uploadY(csvFile("quoted.csv", text));

  expect(await studio.settleUpload()).toBe("ready");
  await expect(studio.alerts).toHaveCount(0);
  await expect(studio.page.getByText(/uploaded Y · quoted\.csv/).first()).toBeVisible();
});

test("accepted upload: a semicolon-separated CSV", async ({ studio }) => {
  const fixture = await smallDesign(studio);
  const text = yCsv(fixture).replaceAll(",", ";");

  await studio.uploadY(csvFile("semi.csv", text));

  expect(await studio.settleUpload()).toBe("ready");
  await expect(studio.alerts).toHaveCount(0);
  await expect(studio.page.getByText(/uploaded Y · semi\.csv/).first()).toBeVisible();
});
