import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid } from "recharts";
import type { PredictResponse } from "../../types/api";

interface ROMComparisonChartProps {
  prediction: PredictResponse;
}

/** Compares real hip_rom vs knee_rom (degrees) from /predict. */
export default function ROMComparisonChart({ prediction }: ROMComparisonChartProps) {
  const data = [
    { joint: "Hip", rom: prediction.hip_rom },
    { joint: "Knee", rom: prediction.knee_rom },
  ];

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: -10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
        <XAxis dataKey="joint" tick={{ fill: "var(--text-light)", fontSize: 12 }} />
        <YAxis unit="°" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
        <Tooltip
          formatter={(value: number) => [`${value.toFixed(1)}°`, "Range of Motion"]}
          contentStyle={{ background: "var(--surface)", border: "1px solid var(--border-color)", borderRadius: 8 }}
        />
        <Bar dataKey="rom" fill="var(--accent)" radius={[6, 6, 0, 0]} isAnimationActive barSize={60} />
      </BarChart>
    </ResponsiveContainer>
  );
}
