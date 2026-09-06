import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { parseCsv, type ParsedCsv } from "./engine";

export interface CsvUploadProps {
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
 * A labeled file input that parses a CSV on selection and hands the parsed
 * rows to the parent. Optional "load example" button variant.
 */
export function CsvUpload({
  label,
  hint,
  disabled,
  onLoaded,
  onError,
  exampleLabel,
  onExample,
  exampleDisabled,
}: CsvUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleFile = async (file: File | undefined) => {
    if (!file) return;
    try {
      const parsed = parseCsv(await file.text());
      setFileName(file.name);
      onLoaded(parsed);
    } catch (err) {
      setFileName(null);
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
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
        {fileName ?? hint ?? "no file loaded"}
      </p>
    </div>
  );
}