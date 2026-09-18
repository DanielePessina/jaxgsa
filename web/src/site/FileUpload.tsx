import { useRef, useState } from "react";
import { FileCheck2, UploadCloud } from "lucide-react";
import { Button } from "@/components/ui/button";
import { parseUpload } from "./upload";
import type { ParsedCsv } from "./engine";

export interface FileUploadProps {
  label: string;
  hint?: string;
  disabled?: boolean;
  onLoaded: (parsed: ParsedCsv) => void;
  onError: (message: string) => void;
  exampleLabel?: string;
  onExample?: () => void;
  exampleDisabled?: boolean;
}

/**
 * A labeled file input that accepts CSV or Parquet, parses it on selection
 * (via `parseUpload`, which rejects unsafe formats like pickle) and hands the
 * shared `ParsedCsv` shape to the parent. Optional "load example" button.
 */
export function FileUpload({
  label,
  hint,
  disabled,
  onLoaded,
  onError,
  exampleLabel,
  onExample,
  exampleDisabled,
}: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [fileLabel, setFileLabel] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = async (file: File | undefined) => {
    if (!file) return;

    try {
      const parsed = await parseUpload(file);
      setFileLabel(`${file.name} · ${parsed.rows.length} rows × ${parsed.headers.length} cols`);
      onLoaded(parsed);
    } catch (err) {
      setFileLabel(null);
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="min-w-[260px] flex-1 space-y-2">
      <div>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.parquet,.pq,text/csv,application/vnd.apache.parquet"
          className="hidden"
          disabled={disabled}
          onChange={(e) => {
            void handleFile(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
        <button
          type="button"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
          onDragEnter={(e) => {
            e.preventDefault();

            if (!disabled) setDragging(true);
          }}
          onDragOver={(e) => e.preventDefault()}
          onDragLeave={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
              setDragging(false);
            }
          }}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);

            if (!disabled) void handleFile(e.dataTransfer.files[0]);
          }}
          className={`flex w-full items-center gap-3 rounded-md border border-dashed px-3 py-3 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-45 ${
            dragging
              ? "border-primary bg-primary/10"
              : "border-border bg-background/35 hover:border-primary/45 hover:bg-muted/25"
          }`}
        >
          <span className={`flex size-8 shrink-0 items-center justify-center rounded-md border ${fileLabel ? "border-primary/35 bg-primary/10 text-primary" : "border-border bg-muted/30 text-muted-foreground"}`}>
            {fileLabel ? <FileCheck2 className="size-4" /> : <UploadCloud className="size-4" />}
          </span>
          <span className="min-w-0">
            <span className="block text-sm font-medium">{fileLabel ? fileLabel : label}</span>
            <span className="mt-0.5 block truncate text-xs text-muted-foreground">
              {fileLabel ? "Click or drop another file to replace it" : "Choose or drop CSV / Parquet"}
            </span>
          </span>
        </button>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="font-mono text-[11px] leading-relaxed text-muted-foreground">
          {hint ?? "CSV or Parquet"}
        </p>
        {exampleLabel && onExample && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={exampleDisabled}
            onClick={onExample}
          >
            {exampleLabel}
          </Button>
        )}
      </div>
    </div>
  );
}
