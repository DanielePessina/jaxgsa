import { describe, expect, it } from "vitest";
import { parquetWriteBuffer } from "hyparquet-writer";
import {
  MAX_CELLS,
  fileExtension,
  isParquetMagic,
  parseUpload,
  rejectUnsafeName,
} from "./upload";

describe("fileExtension", () => {
  it("returns the lowercased extension with the dot", () => {
    expect(fileExtension("run.csv")).toBe(".csv");
    expect(fileExtension("RUN.PARQUET")).toBe(".parquet");
    expect(fileExtension("noext")).toBe("");
    expect(fileExtension("trailing.")).toBe("");
  });
});

describe("rejectUnsafeName", () => {
  it("rejects pickle-family formats with a code-execution reason", () => {
    for (const name of ["out.pkl", "out.pkl.gz", "model.joblib"]) {
      const reason = rejectUnsafeName(name);
      expect(reason).not.toBeNull();
      expect(reason).toMatch(/safely/i);
    }
  });

  it("rejects array and binary formats without column names", () => {
    expect(rejectUnsafeName("y.npy")).toMatch(/no column names/);
    expect(rejectUnsafeName("y.npz")).toMatch(/no column names/);
    expect(rejectUnsafeName("y.pt")).not.toBeNull();
    expect(rejectUnsafeName("y.h5")).not.toBeNull();
  });

  it("accepts CSV and Parquet", () => {
    expect(rejectUnsafeName("design.csv")).toBeNull();
    expect(rejectUnsafeName("design.parquet")).toBeNull();
    expect(rejectUnsafeName("design.pq")).toBeNull();
  });
});

describe("isParquetMagic", () => {
  it("detects the PAR1 magic", () => {
    expect(isParquetMagic(new Uint8Array([0x50, 0x41, 0x52, 0x31]))).toBe(true);
    expect(isParquetMagic(new Uint8Array([0x50, 0x41, 0x52, 0x32]))).toBe(false);
    expect(isParquetMagic(new Uint8Array([0x50, 0x41, 0x52]))).toBe(false);
  });
});

describe("parseUpload", () => {
  function csvFile(name: string, text: string): File {
    return new File([text], name, { type: "text/csv" });
  }

  it("parses a plain CSV", async () => {
    const parsed = await parseUpload(csvFile("y.csv", "run_id,y\n0,1.5\n1,2.5\n"));
    expect(parsed.headers).toEqual(["run_id", "y"]);
    expect(parsed.rows).toEqual([
      [0, 1.5],
      [1, 2.5],
    ]);
  });

  it("parses a misnamed CSV (no extension) by content", async () => {
    const parsed = await parseUpload(csvFile("outputs", "run_id,y\n0,1\n"));
    expect(parsed.headers).toEqual(["run_id", "y"]);
  });

  it("rejects unsafe formats with the user-facing reason", async () => {
    await expect(parseUpload(csvFile("out.pkl", "x"))).rejects.toThrow(/safely/);
    await expect(parseUpload(csvFile("y.npy", "x"))).rejects.toThrow(/no column names/);
  });

  it("parses a real Parquet file", async () => {
    const buf = parquetWriteBuffer({
      columnData: [
        { name: "run_id", data: [1, 0, 2], type: "INT32" },
        { name: "y", data: [3.3, 1.1, 5.5], type: "DOUBLE" },
      ],
    });
    const parsed = await parseUpload(
      new File([buf], "outputs.parquet", { type: "application/vnd.apache.parquet" }),
    );
    expect(parsed.headers).toEqual(["run_id", "y"]);
    expect(parsed.rows).toEqual([
      [1, 3.3],
      [0, 1.1],
      [2, 5.5],
    ]);
  });

  it("parses a Parquet file even when the extension is wrong (magic sniff)", async () => {
    const buf = parquetWriteBuffer({
      columnData: [
        { name: "y", data: [1, 2], type: "INT32" },
      ],
    });
    const parsed = await parseUpload(new File([buf], "outputs.csv"));
    expect(parsed.headers).toEqual(["y"]);
    expect(parsed.rows).toEqual([[1], [2]]);
  });

  it("rejects parquet cells that are not numbers", async () => {
    const buf = parquetWriteBuffer({
      columnData: [
        { name: "run_id", data: [0, 1], type: "INT32" },
        { name: "y", data: ["one", "two"], type: "STRING" },
      ],
    });
    await expect(
      parseUpload(new File([buf], "outputs.parquet")),
    ).rejects.toThrow(/not a number/);
  });

  it("enforces the cell-count guard", async () => {
    const cols = 10;
    const rows = Math.ceil(MAX_CELLS / cols) + 1;
    const header = Array.from({ length: cols }, (_, j) => `c${j}`).join(",");
    const body = Array.from({ length: rows }, () => header.replaceAll(/\w/g, "1")).join("\n");
    await expect(
      parseUpload(csvFile("big.csv", `${header}\n${body}\n`)),
    ).rejects.toThrow(/too large/);
  });
});