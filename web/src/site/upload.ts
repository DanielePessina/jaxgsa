/**
 * Secure upload handling (web/docs/PLAN-WEB-UI.md D2): format detection, unsafe-format
 * rejection, size guards, and CSV/Parquet parsing to the shared `ParsedCsv`
 * shape the engine consumes.
 *
 * Everything is parsed client-side; nothing leaves the page. The only "risk"
 * is parsing hostile bytes, which CSV and Parquet are safe against. Pickle
 * and friends (.pkl/.joblib) can execute arbitrary code on load and are
 * rejected outright; binary array formats without column names (.npy/.npz)
 * are rejected as unusable.
 */

import { parquetMetadata, parquetReadObjects } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import { MAX_CELLS, parseCsv, parseDecimal, type ParsedCsv } from "./engine";

/** Human cap on the number of cells we parse (rows x cols) before bailing. */
export { MAX_CELLS };

function tooLarge(rows: number, cols: number, maxCells: number): Error {
  return new Error(
    `file is too large: ${rows} rows x ${cols} ` +
      `columns exceeds the ${maxCells.toLocaleString()} cell limit`,
  );
}

/** Hard cap on the raw file size (bytes). */
export const MAX_FILE_BYTES = 512 * 1024 * 1024;

const PARQUET_MAGIC = new Uint8Array([0x50, 0x41, 0x52, 0x31]); // "PAR1"

/**
 * Formats we refuse, keyed by lowercased extension, with the reason shown to
 * the user. Pickle-family formats are a code-execution hazard; array formats
 * carry no column names and would be silently misread.
 */
export const UNSAFE_EXTENSIONS = {
  ".pkl": "pickle cannot be loaded safely (it executes arbitrary code); use CSV or Parquet",
  ".pkl.gz": "pickle cannot be loaded safely (it executes arbitrary code); use CSV or Parquet",
  ".joblib": "joblib/pickle cannot be loaded safely (it executes arbitrary code); use CSV or Parquet",
  ".npy": "NumPy .npy carries no column names; use CSV or Parquet",
  ".npz": "NumPy .npz carries no column names; use CSV or Parquet",
  ".pt": "PyTorch serialized tensors are not supported; use CSV or Parquet",
  ".pth": "PyTorch serialized tensors are not supported; use CSV or Parquet",
  ".h5": "HDF5 is not supported; use CSV or Parquet",
  ".hdf5": "HDF5 is not supported; use CSV or Parquet",
  ".mat": "MATLAB is not supported; use CSV or Parquet",
  ".sav": "SPSS is not supported; use CSV or Parquet",
  ".dta": "Stata is not supported; use CSV or Parquet",
  ".xlsx": "Excel is not supported; export a CSV or Parquet instead",
  ".xls": "Excel is not supported; export a CSV or Parquet instead",
} satisfies Record<string, string>;

function unsafeMessage(extension: string): string | undefined {
  return Object.entries(UNSAFE_EXTENSIONS).find(([key]) => key === extension)?.[1];
}

/** Lowercased file extension including the dot, or "" when none. */
export function fileExtension(name: string): string {
  const dot = name.lastIndexOf(".");

  if (dot < 0 || dot === name.length - 1) return "";

  return name.slice(dot).toLowerCase();
}

/** Reason a file name should be refused, or null when it is acceptable. */
export function rejectUnsafeName(name: string): string | null {
  const ext = fileExtension(name);
  const direct = unsafeMessage(ext);

  if (direct) return direct;

  // Multi-part names like "out.pkl.gz": look under the gzip wrapper too.
  if (ext === ".gz") {
    const inner = unsafeMessage(fileExtension(name.slice(0, -3)));

    if (inner) return inner;
  }

  return null;
}

/** True when the first 4 bytes are the Parquet "PAR1" magic. */
export function isParquetMagic(bytes: Uint8Array): boolean {
  if (bytes.length < 4) return false;

  for (let i = 0; i < 4; i++) if (bytes[i] !== PARQUET_MAGIC[i]) return false;

  return true;
}

type ParquetCell = number | bigint | boolean | string | null | undefined;

/** Coerce one parquet cell to a finite number, matching CSV tolerance. */
function cellToNumber(name: string, row: number, v: ParquetCell): number {
  const kind = Object.prototype.toString.call(v);
  const where = `row ${row + 1}, column "${name}"`;

  if (kind === "[object Number]" || kind === "[object BigInt]") {
    const n = Number(v);

    if (!Number.isFinite(n)) throw new Error(`${where}: ${String(v)} is not a finite number`);

    return n;
  }

  if (kind === "[object Boolean]") return v === true ? 1 : 0;

  if (kind === "[object String]" && String(v).trim() !== "") {
    const parsed = parseDecimal(String(v));

    if ("value" in parsed) return parsed.value;

    throw new Error(`${where}: ${String(v)} is ${parsed.reason}`);
  }

  throw new Error(`${where}: ${String(v)} is not a number`);
}

/** Parse a Parquet file (with full codec support) into the shared CSV shape. */
async function parseParquet(file: File, maxCells: number): Promise<ParsedCsv> {
  const buf = await file.arrayBuffer();

  // Check the size from the footer before any row is decoded.
  try {
    const meta = parquetMetadata(buf);
    const cols = Math.max(meta.schema.length - 1, 1);
    const rowCount = Number(meta.num_rows);

    if (rowCount * cols > maxCells) throw tooLarge(rowCount, cols, maxCells);
  } catch (err) {
    if (err instanceof Error && err.message.startsWith("file is too large")) throw err;

    throw new Error(
      `could not read parquet: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  let rows;

  try {
    rows = await parquetReadObjects({ file: buf, compressors });
  } catch (err) {
    throw new Error(
      `could not read parquet: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  if (rows.length === 0) {
    throw new Error("the parquet file has no data rows");
  }

  const headers = Object.keys(rows[0]);
  const out: number[][] = [];

  for (let r = 0; r < rows.length; r++) {
    const row = rows[r];
    const cells: number[] = [];

    for (const h of headers) {
      const v = row[h];

      if (v === null || v === undefined) {
        throw new Error(`row ${r + 1}, column "${h}": empty cell`);
      }

      cells.push(cellToNumber(h, r, v));
    }

    out.push(cells);
  }

  return { headers, rows: out };
}

/**
 * Parse an uploaded file into the shared `ParsedCsv` shape. Detects the
 * format by extension first, then by magic bytes (so a misnamed parquet still
 * parses). Rejects unsafe formats and over-size inputs.
 */
export async function parseUpload(
  file: File,
  maxCells: number = MAX_CELLS,
): Promise<ParsedCsv> {
  if (file.size > MAX_FILE_BYTES) {
    throw new Error(
      `file is ${(file.size / 1024 / 1024).toFixed(1)} MB — the limit is ` +
        `${Math.round(MAX_FILE_BYTES / 1024 / 1024)} MB`,
    );
  }

  const unsafe = rejectUnsafeName(file.name);

  if (unsafe) throw new Error(`${file.name}: ${unsafe}`);

  const ext = fileExtension(file.name);

  if (ext === ".parquet" || ext === ".pq") {
    return parseParquet(file, maxCells);
  }

  const head = new Uint8Array(await file.slice(0, 4).arrayBuffer());

  if (isParquetMagic(head)) return parseParquet(file, maxCells);

  // `parseCsv` rejects an oversize file from the line count, before it builds any row.
  return parseCsv(await file.text(), maxCells);
}
