"use client";

import { useEffect, useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from "recharts";

interface FrogChartProps {
  cameraIds: number[];                
  graphData: Array<GraphData> | null | undefined;
}

export interface GraphData {
  runDate: string;                     // e.g., "2025-10-10T12_30_00"
  frogs: number;
  camera: number;
}

type ChartRow = {
  month: string;                       // yyyy-mm-dd
  [seriesKey: `camera${number}`]: number | null;
};

export default function FrogChart({ cameraIds, graphData }: FrogChartProps) {
  const [data, setData] = useState<ChartRow[]>([]);

  // simple palette: accent first, then fallbacks
  const colorPalette = [
    "var(--apple-accent)",
    "#8884d8",
    "#82ca9d",
    "#ff7300",
    "#00C49F",
    "#FF8042",
    "#A28CF6",
    "#4DD0E1",
  ];

  // Build a date index across selected cameras, then one value per camera per date
  const seriesKeys = useMemo(() => cameraIds.map((c) => `camera${c}` as const), [cameraIds]);

  useEffect(() => {
    if (!graphData || cameraIds.length === 0) {
      setData([]);
      return;
    }

    // filter to selected cameras
    const filtered = graphData.filter(d => cameraIds.includes(d.camera));

    // collect unique dates (yyyy-mm-dd) across all selected cameras
    const allDates = Array.from(
      new Set(filtered.map(d => d.runDate.split("T")[0]))
    ).sort((a, b) => new Date(a).getTime() - new Date(b).getTime());

    // create a lookup: {date -> {cameraN: value}}
    const byDate: Record<string, Partial<ChartRow>> = {};
    for (const date of allDates) {
      byDate[date] = { month: date };
      for (const cam of cameraIds) {
        (byDate[date] as any)[`camera${cam}`] = null; // fill later
      }
    }

    // fill values
    for (const row of filtered) {
      const date = row.runDate.split("T")[0];
      const key = `camera${row.camera}` as const;
      const current = (byDate[date] as any)[key];
      (byDate[date] as any)[key] = (current ?? 0) + row.frogs;
    }

    // finalize array
    const chartRows = allDates.map(d => byDate[d] as ChartRow);
    setData(chartRows);
  }, [graphData, cameraIds]);

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{
            top: 20,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" vertical={false} />
          <XAxis
            dataKey="month"
            tick={{ fill: "var(--apple-text-secondary)" }}
            axisLine={{ stroke: "#e0e0e0" }}
            tickLine={{ stroke: "#e0e0e0" }}
          />
          <YAxis
            tick={{ fill: "var(--apple-text-secondary)" }}
            axisLine={{ stroke: "#e0e0e0" }}
            tickLine={{ stroke: "#e0e0e0" }}
            label={{
              value: "frogs",
              angle: -90,
              position: "insideLeft",
              fill: "var(--apple-text-secondary)",
              dx: -10
            }}
            allowDecimals={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--apple-card-bg)",
              border: "1px solid var(--apple-border)",
              borderRadius: "8px",
              boxShadow: "0 4px 12px var(--apple-shadow)",
              color: "var(--apple-text)"
            }}
          />
          <Legend />
          <ReferenceLine y={40} stroke="#e0e0e0" strokeDasharray="3 3" />

          {cameraIds.map((cam, idx) => {
            const series = `camera${cam}` as const;
            return (
              <Line
                key={series}
                type="monotone"
                dataKey={series}
                name={`Camera ${cam}`}
                stroke={colorPalette[idx % colorPalette.length]}
                strokeWidth={3}
                dot={{ fill: "#fff", stroke: colorPalette[idx % colorPalette.length], strokeWidth: 2, r: 5 }}
                activeDot={{ r: 7, fill: colorPalette[idx % colorPalette.length] }}
                connectNulls
                animationDuration={800}
                animationEasing="ease-out"
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
