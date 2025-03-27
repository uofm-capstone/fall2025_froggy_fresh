"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from "recharts";

interface FrogChartProps {
  cameraId: string;
}

export default function FrogChart({ cameraId }: FrogChartProps) {
  const [data, setData] = useState<any[]>([]);

  // Generate different data based on camera ID
  useEffect(() => {
    // Base data
    const baseData = [
      { month: "Feb", frogs: 20 },
      { month: "Mar", frogs: 22 },
      { month: "Apr", frogs: 28 },
      { month: "May", frogs: 40 },
      { month: "Jun", frogs: 52 },
      { month: "Jul", frogs: 68 },
      { month: "Aug", frogs: 75 },
      { month: "Sep", frogs: 80 },
      { month: "Oct", frogs: 78 },
      { month: "Nov", frogs: 80 },
      { month: "Dec", frogs: 80 },
    ];

    // Generate different data for each camera
    let chartData = [...baseData];
    if (cameraId === "camera2") {
      chartData = baseData.map(point => ({
        ...point,
        frogs: Math.floor(point.frogs * 0.8)
      }));
    } else if (cameraId === "camera3") {
      chartData = baseData.map(point => ({
        ...point,
        frogs: Math.floor(point.frogs * 1.2)
      }));
    }

    setData(chartData);
  }, [cameraId]);

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
          <ReferenceLine y={40} stroke="#e0e0e0" strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="frogs"
            stroke="var(--apple-accent)"
            strokeWidth={3}
            dot={{ fill: "#fff", stroke: "var(--apple-accent)", strokeWidth: 2, r: 5 }}
            activeDot={{ r: 8, fill: "var(--apple-accent)" }}
            animationDuration={1000}
            animationEasing="ease-out"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
