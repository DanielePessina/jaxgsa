/**
 * Results visualization (PLAN-WEB-UI.md D3/D4): grouped bar chart of the
 * sensitivity indices behind an isolated component so the chart library can
 * be swapped (AG Charts today, Reaviz later) without touching the cards.
 *
 * Editorial-scientific styling: transparent background, Geist Mono, muted
 * gridlines, restrained palette. The "matplotlib look" (minor gridlines +
 * minor ticks, sparse major labels) is emulated with a fine `interval.step`
 * and a label formatter that only renders labels at major multiples.
 */

import { Component, useMemo, useRef, type ErrorInfo, type ReactNode } from "react";
import { AgCharts } from "ag-charts-react";
import type {
  AgCartesianChartOptions,
  AgChartInstance,
  AgChartTheme,
} from "ag-charts-community";
import { AllCommunityModule, ModuleRegistry } from "ag-charts-community";
import { Button } from "@/components/ui/button";
import { downloadText } from "./download";
import type { AnalysisResult } from "./engine";

// AG Charts 14 no longer self-registers its chart modules. Register the
// community bundle once before the first chart mounts; without this the
// results view throws at runtime and React removes the whole workbench.
ModuleRegistry.registerModules([AllCommunityModule]);

/** Series palette from the site's dark chart tokens (index.css `.dark`). */
const PALETTE = [
  "oklch(0.72 0.14 195)",
  "oklch(0.62 0.16 250)",
  "oklch(0.7 0.14 60)",
  "oklch(0.65 0.13 320)",
  "oklch(0.68 0.12 150)",
];

const GRIDLINE = "oklch(0.63 0.012 250 / 22%)";
const TICK_COLOR = "oklch(0.63 0.012 250 / 60%)";
const LABEL_COLOR = "oklch(0.63 0.012 250)";

/**
 * Choose a "nice" major step (~4-5 major lines over `span`) and a minor step
 * one fifth of it, so the minor grid is dense but the labels stay sparse.
 */
export function niceSteps(span: number): { major: number; minor: number } {
  if (!(span > 0) || !Number.isFinite(span)) return { major: 0.2, minor: 0.04 };
  const rough = span / 5;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / mag;
  const major = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  return { major, minor: major / 5 };
}

function isMajorTick(v: number, major: number): boolean {
  if (!Number.isFinite(v)) return false;
  const q = v / major;
  return Math.abs(q - Math.round(q)) < 1e-9;
}

const theme: AgChartTheme = {
  baseTheme: "ag-default-dark",
  palette: { fills: PALETTE, strokes: PALETTE },
  params: {
    fontFamily: "Geist Mono",
    fontSize: 11,
    foregroundColor: LABEL_COLOR,
  },
  overrides: {
    common: {
      background: { visible: false, fill: "transparent" },
      legend: { position: "bottom" },
    },
    bar: {
      series: {
        strokeWidth: 0,
        cornerRadius: 2,
      },
    },
  },
};

export interface ChartDataRow {
  parameter: string;
  [indexLabel: string]: string | number;
}

/** Shape one result slice into chart rows (pure, tested). */
export function shapeChartData(
  result: AnalysisResult,
  sliceIndex: number,
): ChartDataRow[] {
  const slice = result.slices[sliceIndex];
  if (!slice) {
    throw new Error(`IndicesChart: no slice ${sliceIndex} in ${result.method} result`);
  }
  return result.parameters.map((p, i) => {
    const row: ChartDataRow = { parameter: p };
    for (const c of slice.columns) row[c.label] = c.values[i];
    return row;
  });
}

function toCsv(result: AnalysisResult, sliceIndex: number): string {
  const slice = result.slices[sliceIndex];
  const rows = shapeChartData(result, sliceIndex);
  const labels = slice.columns.map((c) => c.label);
  const lines = [`parameter,${labels.join(",")}`];
  for (const row of rows) {
    lines.push([row.parameter, ...labels.map((l) => String(row[l]))].join(","));
  }
  return lines.join("\n") + "\n";
}

export function IndicesChart({
  result,
  sliceIndex,
  fileName,
}: {
  result: AnalysisResult;
  sliceIndex: number;
  fileName: string;
}) {
  const chartRef = useRef<AgChartInstance | null>(null);

  const options = useMemo<AgCartesianChartOptions>(() => {
    const slice = result.slices[sliceIndex];
    const data = shapeChartData(result, sliceIndex);
    let max = 0;
    for (const c of slice.columns) {
      for (const v of c.values) if (v > max) max = v;
    }
    max = Math.max(max, 1);
    const { major, minor } = niceSteps(max);

    return {
      data,
      series: slice.columns.map((c, j) => ({
        type: "bar",
        xKey: "parameter",
        yKey: c.label,
        fill: PALETTE[j % PALETTE.length],
        stroke: "transparent",
        strokeWidth: 0,
      })),
      axes: {
        x: {
          type: "category",
          label: { fontFamily: "Geist Mono", fontSize: 11, color: LABEL_COLOR },
          gridLine: { enabled: false },
          tick: { enabled: false },
        },
        y: {
          type: "number",
          min: 0,
          max,
          interval: { step: minor },
          label: {
            fontFamily: "Geist Mono",
            fontSize: 10,
            color: LABEL_COLOR,
            formatter: ({ value }) =>
              isMajorTick(value as number, major) ? String(value) : "",
          },
          tick: { stroke: TICK_COLOR, width: 1 },
          gridLine: { enabled: true, width: 1, style: [{ stroke: GRIDLINE }] },
          crosshair: { enabled: false },
        },
      },
      theme,
      background: { visible: false, fill: "transparent" },
      legend: { position: "bottom" },
    };
  }, [result, sliceIndex]);

  const onPng = () => {
    if (chartRef.current) void chartRef.current.download({ fileName });
  };

  const onCsv = () => {
    downloadText(`${fileName}.csv`, toCsv(result, sliceIndex), "text/csv");
  };

  return (
    <div className="space-y-2">
      <div className="flex justify-end gap-1.5">
        <Button type="button" variant="ghost" size="sm" onClick={onCsv} className="font-mono text-xs">
          data csv
        </Button>
        <Button type="button" variant="ghost" size="sm" onClick={onPng} className="font-mono text-xs">
          png
        </Button>
      </div>
      <div className="min-h-[340px] sm:min-h-[380px]">
        <ChartErrorBoundary>
          <AgCharts ref={chartRef} options={options} />
        </ChartErrorBoundary>
      </div>
    </div>
  );
}

/** Keep a chart integration failure local to its result card. */
class ChartErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, _info: ErrorInfo) {
    console.error("The sensitivity chart could not be rendered.", error);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex min-h-[300px] items-center justify-center rounded-sm border border-dashed border-border/70 px-6 text-center text-sm text-muted-foreground">
          Chart unavailable. The sensitivity table below still contains the
          computed values.
        </div>
      );
    }
    return this.props.children;
  }
}
