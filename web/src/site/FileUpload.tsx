import { useRef, useState } from "react";
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
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
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
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          {label}
        </Button>
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
      <p className="font-mono text-xs text-muted-foreground">
        {fileLabel ?? hint ?? "no file loaded"}
      </p>
    </div>
  );
}