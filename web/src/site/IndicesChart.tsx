import { useId, useMemo, useRef } from "react";
import { extent } from "d3-array";
import { scaleBand, scaleLinear } from "d3-scale";
import { Button } from "@/components/ui/button";
import { downloadText } from "./download";
import { formatIndex } from "./format";
import type { AnalysisResult } from "./engine";

const WIDTH = 720;

const HEIGHT = 400;

const MARGIN = { top: 24, right: 18, bottom: 70, left: 54 };

const PALETTE = ["#5cc3c8", "#4f8fe5", "#d6a25c", "#bd72c7", "#62b68c"];

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

export function shapeChartData(result: AnalysisResult, sliceIndex: number): ChartDataRow[] {
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
  const rows = shapeChartData(result, sliceIndex);
  const labels = slice.columns.map((column) => column.label);
  const lines = [`parameter,${labels.join(",")}`];

  for (const row of rows) {
    lines.push([row.parameter, ...labels.map((label) => String(row[label]))].join(","));
  }

  return lines.join("\n") + "\n";
}

function downloadPng(svg: SVGSVGElement, fileName: string) {
  const source = new XMLSerializer().serializeToString(svg);
  const url = URL.createObjectURL(new Blob([source], { type: "image/svg+xml;charset=utf-8" }));
  const image = new Image();
  image.onload = () => {
    const scale = 2;
    const canvas = document.createElement("canvas");
    canvas.width = WIDTH * scale;
    canvas.height = HEIGHT * scale;
    const ctx = canvas.getContext("2d");

    if (!ctx) return;
    ctx.scale(scale, scale);
    ctx.fillStyle = "#111318";
    ctx.fillRect(0, 0, WIDTH, HEIGHT);
    ctx.drawImage(image, 0, 0, WIDTH, HEIGHT);
    const link = document.createElement("a");
    link.download = `${fileName}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
    URL.revokeObjectURL(url);
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
  const slice = result.slices[sliceIndex];

  const chart = useMemo(() => {
    const data = shapeChartData(result, sliceIndex);
    const values = slice.columns.flatMap((column) => Array.from(column.values));
    const [min, max] = chartDomain(values);

    const x = scaleBand<string>()
      .domain(result.parameters)
      .range([MARGIN.left, WIDTH - MARGIN.right])
      .paddingInner(0.22)
      .paddingOuter(0.08);

    const series = scaleBand<string>()
      .domain(slice.columns.map((column) => column.label))
      .range([0, x.bandwidth()])
      .padding(0.1);

    const y = scaleLinear()
      .domain([min, max])
      .nice(5)
      .range([HEIGHT - MARGIN.bottom, MARGIN.top]);

    return { data, x, series, y, ticks: y.ticks(5), zero: y(0) };
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
          <Button type="button" variant="ghost" size="sm" onClick={() => svgRef.current && downloadPng(svgRef.current, fileName)} className="font-mono text-xs">png</Button>
        </div>
      </div>

      <svg ref={svgRef} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-labelledby={titleId} className="block h-auto w-full" xmlns="http://www.w3.org/2000/svg">
        <title id={titleId}>{result.method} sensitivity indices for {slice.output} at time {slice.time}</title>
        <rect width={WIDTH} height={HEIGHT} fill="#111318" rx="6" />
        {chart.ticks.map((tick) => (
          <g key={tick}>
            <line x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={chart.y(tick)} y2={chart.y(tick)} stroke={tick === 0 ? "#6e7683" : "#2a2e35"} strokeWidth={tick === 0 ? 1.2 : 1} />
            <text x={MARGIN.left - 10} y={chart.y(tick)} dy="0.35em" textAnchor="end" fill="#8d949e" fontFamily="Geist Mono, monospace" fontSize="11">{formatIndex(tick)}</text>
          </g>
        ))}
        {chart.data.flatMap((row) =>
          slice.columns.map((column, seriesIndex) => {
            const value = Number(row[column.label]);
            const x = chart.x(String(row.parameter))! + chart.series(column.label)!;
            const y = value >= 0 ? chart.y(value) : chart.zero;
            const height = Math.max(1, Math.abs(chart.y(value) - chart.zero));

            return (
              <rect key={`${row.parameter}-${column.key}`} x={x} y={y} width={chart.series.bandwidth()} height={height} rx="2" fill={value < 0 ? "#df6b63" : PALETTE[seriesIndex % PALETTE.length]}>
                <title>{row.parameter} · {column.label}: {formatIndex(value)}</title>
              </rect>
            );
          }),
        )}
        {result.parameters.map((parameter) => (
          <text key={parameter} x={chart.x(parameter)! + chart.x.bandwidth() / 2} y={HEIGHT - MARGIN.bottom + 26} textAnchor="middle" fill="#a5abb4" fontFamily="Geist Mono, monospace" fontSize="12">{parameter}</text>
        ))}
      </svg>
    </div>
  );
}
