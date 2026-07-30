import { PieChart, Pie, Cell } from "recharts";
import { useCountUp } from "../../hooks/useCountUp";

interface PerformanceGaugeProps {
  score: number; // 0-100, real confidence_weighted_score from /predict
  label?: string;
}

/** Semi-circle gauge visualizing the real confidence_weighted_score (0-100). */
export default function PerformanceGauge({ score, label = "Performance Score" }: PerformanceGaugeProps) {
  const clamped = Math.min(100, Math.max(0, score));
  const animated = useCountUp(clamped);
  const color = clamped >= 70 ? "var(--good)" : clamped >= 40 ? "var(--warn)" : "var(--bad)";

  const data = [
    { name: "score", value: clamped },
    { name: "rest", value: 100 - clamped },
  ];

  return (
    <div className="performance-gauge text-center">
      <PieChart width={220} height={130}>
        <Pie
          data={data}
          cx={110}
          cy={110}
          startAngle={180}
          endAngle={0}
          innerRadius={70}
          outerRadius={100}
          dataKey="value"
          stroke="none"
          isAnimationActive
        >
          <Cell fill={color} />
          <Cell fill="var(--border-color)" />
        </Pie>
      </PieChart>
      <div className="performance-gauge-value">
        <span className="performance-gauge-number">{animated.toFixed(1)}</span>
        <span className="performance-gauge-max">/100</span>
      </div>
      <span className="performance-gauge-label">{label}</span>
    </div>
  );
}
