import { PieChart, Pie, Cell } from "recharts";
import { useCountUp } from "../../hooks/useCountUp";

interface ConfidenceGaugeProps {
  confidence: number; // 0-1, real prediction_confidence
  lowConfidenceFlag: boolean;
}

/** Full circular gauge visualizing the real prediction_confidence (0-1). */
export default function ConfidenceGauge({ confidence, lowConfidenceFlag }: ConfidenceGaugeProps) {
  const pct = Math.min(100, Math.max(0, confidence * 100));
  const animated = useCountUp(pct);
  const tone = lowConfidenceFlag ? "var(--bad)" : pct >= 75 ? "var(--good)" : "var(--warn)";

  const data = [
    { name: "confidence", value: pct },
    { name: "rest", value: 100 - pct },
  ];

  return (
    <div className="confidence-gauge text-center">
      <PieChart width={140} height={140}>
        <Pie
          data={data}
          cx={70}
          cy={70}
          startAngle={90}
          endAngle={-270}
          innerRadius={50}
          outerRadius={65}
          dataKey="value"
          stroke="none"
          isAnimationActive
        >
          <Cell fill={tone} />
          <Cell fill="var(--border-color)" />
        </Pie>
      </PieChart>
      <div className="confidence-gauge-value">{animated.toFixed(1)}%</div>
      <span className="confidence-gauge-label">{lowConfidenceFlag ? "Low Confidence" : "Confidence"}</span>
    </div>
  );
}
