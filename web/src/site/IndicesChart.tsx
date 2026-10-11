import { useId, useMemo, useRef, useState } from "react";
import { extent } from "d3-array";
import { scaleBand, scaleLinear } from "d3-scale";
import { Button } from "@/components/ui/button";
import { downloadBlob, downloadText } from "./download";
import { formatIndex } from "./format";
import type { AnalysisResult } from "./engine";

const WIDTH = 720;

const HEIGHT = 400;

const MARGIN = { top: 24, right: 18, bottom: 70, left: 54 };

/** Extra bottom margin when the x labels are rotated. */
const ROTATED_BOTTOM = 112;

/** Width of one monospace character at the 12px label size. */
const CHAR_WIDTH = 7.3;

/** Vertical room one rotated label needs, so neighbours do not overlap. */
const ROTATED_STEP = 14;

const MAX_LABEL_CHARS = 16;

/** Space kept between a rotated label and the left edge of the image. */
const LABEL_EDGE_PAD = 6;

/** Fonts for the exported PNG: the web font does not load inside an SVG image. */
const EXPORT_FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

const PALETTE = [
  "var(--chart-bar-1)",
  "var(--chart-bar-2)",
  "var(--chart-bar-3)",
  "var(--chart-bar-4)",
  "var(--chart-bar-5)",
];

export interface ChartDataRow {
  parameter: string;
  [indexLabel: string]: string | number;
}

/** Domain with a visible zero baseline; negative finite estimates are kept. */
export function chartDomain(values: number[]): [number, number] {
  const finite = values.filter(Number.isFinite);
  const [rawMin = 0, rawMax = 1] = extent(finite);
  let min = Math.min(0, rawMin);
  let max = Math.max(0, rawMax);

  if (min === max) max = min + 1;
  const padding = (max - min) * 0.08;

  if (min < 0) min -= padding;
  max += padding;

  return [min, max];
}

export interface BarGeometry {
  y: number;
  height: number;
}

/**
 * Vertical placement of one bar, or `null` when the value is not a finite
 * number (NaN or +-Infinity). The chart draws an "n/a" marker for `null`
 * instead of a rect with NaN attributes.
 */
export function barGeometry(
  value: number,
  yOf: (v: number) => number,
  zero: number,
): BarGeometry | null {
  if (!Number.isFinite(value)) return null;

  const top = yOf(value);

  if (!Number.isFinite(top) || !Number.isFinite(zero)) return null;

  return {
    y: value >= 0 ? top : zero,
    height: Math.max(1, Math.abs(top - zero)),
  };
}

export interface LabelLayout {
  rotate: boolean;
  /** Labels longer than this are cut with an ellipsis. */
  maxChars: number;
  /** Show every `stride`-th label (1 shows all). */
  stride: number;
  /** Bottom margin of the plot, in viewBox units. */
  bottom: number;
  /** Left margin of the plot, in viewBox units. Wider when rotated labels reach past the axis. */
  left: number;
}

/**
 * Decide how to draw the x labels. Labels stay flat while they fit their
 * band. Past that they rotate and are cut short, and when even rotated
 * labels would touch, only every `stride`-th label is drawn. The table below
 * the chart always lists every parameter.
 */
export function labelLayout(parameters: string[], plotWidth: number): LabelLayout {
  const count = Math.max(parameters.length, 1);
  const step = plotWidth / count;
  const longest = parameters.reduce((m, p) => Math.max(m, p.length), 0);

  if (longest * CHAR_WIDTH <= step * 0.92) {
    return { rotate: false, maxChars: longest, stride: 1, bottom: MARGIN.bottom, left: MARGIN.left };
  }

  // A rotated label ends at its band centre and slopes up and to the left at
  // 45 degrees. The first label reaches furthest past the plot edge.
  const first = Math.min(parameters[0]?.length ?? 0, MAX_LABEL_CHARS);
  const reach = first * CHAR_WIDTH * Math.SQRT1_2 - step / 2 + LABEL_EDGE_PAD;

  return {
    rotate: true,
    maxChars: MAX_LABEL_CHARS,
    stride: Math.max(1, Math.ceil(ROTATED_STEP / step)),
    bottom: ROTATED_BOTTOM,
    left: Math.ceil(Math.max(MARGIN.left, reach)),
  };
}

/** Cut a label to `max` characters with an ellipsis. */
export function shortenLabel(label: string, max: number): string {
  return label.length <= max ? label : `${label.slice(0, Math.max(1, max - 1))}…`;
}

/**
 * Replace every `var(--name)` in serialized SVG text with its computed
 * value, so the image does not depend on the page's stylesheet. A name with
 * no value is replaced by `fallback`.
 */
export function resolveCssVars(
  source: string,
  lookup: (name: string) => string,
  fallback = "currentColor",
): string {
  return source.replace(/var\((--[\w-]+)\)/g, (_, name: string) => lookup(name) || fallback);
}

export function chartRowsForSlice(result: AnalysisResult, sliceIndex: number): ChartDataRow[] {
  const slice = result.slices[sliceIndex];

  if (!slice) throw new Error(`IndicesChart: no slice ${sliceIndex} in ${result.method} result`);

  return result.parameters.map((parameter, i) => {
    const row: ChartDataRow = { parameter };

    for (const column of slice.columns) row[column.label] = column.values[i];

    return row;
  });
}

function toCsv(result: AnalysisResult, sliceIndex: number): string {
  const slice = result.slices[sliceIndex];
  const rows = chartRowsForSlice(result, sliceIndex);
  const labels = slice.columns.map((column) => column.label);
  const lines = [`parameter,${labels.join(",")}`];

  for (const row of rows) {
    lines.push([row.parameter, ...labels.map((label) => String(row[label]))].join(","));
  }

  return lines.join("\n") + "\n";
}

/**
 * Serialise a clone of the chart with explicit `width` and `height`. The live
 * SVG has only a viewBox, and Firefox cannot draw such an image to a canvas.
 */
export function serializeForExport<T>(svg: { cloneNode: (deep: boolean) => T }): string {
  const clone = svg.cloneNode(true);

  if (!(clone instanceof Element)) {
    throw new Error("IndicesChart: the chart node cannot take attributes");
  }

  clone.setAttribute("width", String(WIDTH));
  clone.setAttribute("height", String(HEIGHT));

  return new XMLSerializer().serializeToString(clone);
}

function downloadPng(svg: SVGSVGElement, fileName: string, onError: (message: string) => void) {
  const fail = () => onError("Could not create the PNG. Download the data CSV instead.");
  const colors = getComputedStyle(document.documentElement);
  const lookup = (name: string) => colors.getPropertyValue(name).trim();

  // Bake the theme colours and a system font into the image: an SVG drawn
  // as an image sees neither the page stylesheet nor the page's web fonts.
  const source = resolveCssVars(serializeForExport(svg), lookup)
    .replaceAll("Geist Mono, monospace", EXPORT_FONT);

  const url = URL.createObjectURL(new Blob([source], { type: "image/svg+xml;charset=utf-8" }));
  const image = new Image();

  image.onload = () => {
    try {
      const scale = 2;
      const canvas = document.createElement("canvas");
      canvas.width = WIDTH * scale;
      canvas.height = HEIGHT * scale;
      const ctx = canvas.getContext("2d");

      if (!ctx) {
        fail();

        return;
      }

      ctx.scale(scale, scale);
      ctx.fillStyle = lookup("--chart-surface") || "#ffffff";
      ctx.fillRect(0, 0, WIDTH, HEIGHT);
      ctx.drawImage(image, 0, 0, WIDTH, HEIGHT);

      canvas.toBlob((blob) => {
        if (blob) downloadBlob(`${fileName}.png`, blob);
        else fail();
      }, "image/png");
    } catch {
      fail();
    } finally {
      URL.revokeObjectURL(url);
    }
  };

  image.onerror = () => {
    URL.revokeObjectURL(url);
    fail();
  };

  image.src = url;
}

export function IndicesChart({ result, sliceIndex, fileName }: {
  result: AnalysisResult;
  sliceIndex: number;
  fileName: string;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const titleId = useId();
  const [exportError, setExportError] = useState<string | null>(null);
  const slice = result.slices[sliceIndex];

  const chart = useMemo(() => {
    const data = chartRowsForSlice(result, sliceIndex);
    const values = slice.columns.flatMap((column) => Array.from(column.values));
    const [min, max] = chartDomain(values);

    const labels = labelLayout(result.parameters, WIDTH - MARGIN.left - MARGIN.right);
    const left = labels.left;

    const x = scaleBand<string>()
      .domain(result.parameters)
      .range([left, WIDTH - MARGIN.right])
      .paddingInner(0.22)
      .paddingOuter(0.08);

    const series = scaleBand<string>()
      .domain(slice.columns.map((column) => column.label))
      .range([0, x.bandwidth()])
      .padding(0.1);

    const y = scaleLinear()
      .domain([min, max])
      .nice(5)
      .range([HEIGHT - labels.bottom, MARGIN.top]);

    return { data, x, series, y, labels, left, ticks: y.ticks(5), zero: y(0) };
  }, [result, slice, sliceIndex]);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3" aria-label="chart legend">
          {slice.columns.map((column, i) => (
            <span key={column.key} className="flex items-center gap-1.5 font-mono text-[11px] text-muted-foreground">
              <span className="size-2.5 rounded-[2px]" style={{ background: PALETTE[i % PALETTE.length] }} />
              {column.label}
            </span>
          ))}
        </div>
        <div className="flex gap-1.5">
          <Button type="button" variant="ghost" size="sm" onClick={() => downloadText(`${fileName}.csv`, toCsv(result, sliceIndex), "text/csv")} className="font-mono text-xs">data csv</Button>
          <Button type="button" variant="ghost" size="sm" onClick={() => {
            setExportError(null);

            if (svgRef.current) downloadPng(svgRef.current, fileName, setExportError);
          }} className="font-mono text-xs">png</Button>
        </div>
      </div>

      <svg ref={svgRef} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-labelledby={titleId} className="block h-auto w-full" xmlns="http://www.w3.org/2000/svg">
        <title id={titleId}>{result.method} sensitivity indices for {slice.output} at time {slice.time}. The table below the chart lists every value.</title>
        <rect width={WIDTH} height={HEIGHT} fill="var(--chart-surface)" rx="6" />
        {chart.ticks.map((tick) => (
          <g key={tick}>
            <line x1={chart.left} x2={WIDTH - MARGIN.right} y1={chart.y(tick)} y2={chart.y(tick)} stroke={tick === 0 ? "var(--chart-zero)" : "var(--chart-grid)"} strokeWidth={tick === 0 ? 1.2 : 1} />
            <text x={chart.left - 10} y={chart.y(tick)} dy="0.35em" textAnchor="end" fill="var(--chart-tick)" fontFamily="Geist Mono, monospace" fontSize="11">{formatIndex(tick)}</text>
          </g>
        ))}
        {chart.data.flatMap((row) =>
          slice.columns.map((column, seriesIndex) => {
            const value = Number(row[column.label]);
            const x = chart.x(String(row.parameter))! + chart.series(column.label)!;
            const bar = barGeometry(value, chart.y, chart.zero);

            if (!bar) {
              // NaN or Infinity: no bar, a gap with an n/a marker on the baseline.
              return (
                <text key={`${row.parameter}-${column.key}`} x={x + chart.series.bandwidth() / 2} y={chart.zero - 4} textAnchor="middle" fill="var(--chart-tick)" fontFamily="Geist Mono, monospace" fontSize="9">
                  n/a
                  <title>{row.parameter} · {column.label}: {formatIndex(value)} (not a finite number)</title>
                </text>
              );
            }

            return (
              <rect key={`${row.parameter}-${column.key}`} x={x} y={bar.y} width={chart.series.bandwidth()} height={bar.height} rx="2" fill={value < 0 ? "var(--chart-negative)" : PALETTE[seriesIndex % PALETTE.length]}>
                <title>{row.parameter} · {column.label}: {formatIndex(value)}</title>
              </rect>
            );
          }),
        )}
        {result.parameters.map((parameter, i) => {
          if (i % chart.labels.stride !== 0) return null;

          const cx = chart.x(parameter)! + chart.x.bandwidth() / 2;
          const cy = HEIGHT - chart.labels.bottom + 18;
          const text = shortenLabel(parameter, chart.labels.maxChars);

          return (
            <text
              key={parameter}
              x={cx}
              y={cy}
              textAnchor={chart.labels.rotate ? "end" : "middle"}
              transform={chart.labels.rotate ? `rotate(-45 ${cx} ${cy})` : undefined}
              fill="var(--chart-label)"
              fontFamily="Geist Mono, monospace"
              fontSize="12"
            >
              {text}
              {text !== parameter && <title>{parameter}</title>}
            </text>
          );
        })}
      </svg>
      {exportError && (
        <p role="alert" className="text-xs text-destructive">
          {exportError}
        </p>
      )}
    </div>
  );
}
