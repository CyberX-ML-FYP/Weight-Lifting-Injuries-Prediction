import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid } from "recharts";
import type { PredictResponse } from "../../types/api";

interface VelocityComparisonChartProps {
  prediction: PredictResponse;
}

/** Compares real, anthropometrically-normalized peak hip vs. knee angular velocity from /predict. */
export default function VelocityComparisonChart({ prediction }: VelocityComparisonChartProps) {
  const data = [
    { joint: "Hip", velocity: prediction.anthropometric_metrics.hip_peak_velocity_normalized },
    { joint: "Knee", velocity: prediction.anthropometric_metrics.knee_peak_velocity_normalized },
  ];

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: -10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
        <XAxis dataKey="joint" tick={{ fill: "var(--text-light)", fontSize: 12 }} />
        <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
        <Tooltip
          formatter={(value: number) => [`${value.toFixed(3)} leg-lengths/s`, "Peak Velocity"]}
          contentStyle={{ background: "var(--surface)", border: "1px solid var(--border-color)", borderRadius: 8 }}
        />
        <Bar dataKey="velocity" fill="var(--accent-orange)" radius={[6, 6, 0, 0]} isAnimationActive barSize={60} />
      </BarChart>
    </ResponsiveContainer>
  );
}
