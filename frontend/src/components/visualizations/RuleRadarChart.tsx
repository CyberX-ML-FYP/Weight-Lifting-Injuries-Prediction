import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from "recharts";
import type { PredictResponse } from "../../types/api";

interface RuleRadarChartProps {
  prediction: PredictResponse;
}

/** Radar chart over the four real Rule A-D scores. */
export default function RuleRadarChart({ prediction }: RuleRadarChartProps) {
  const data = [
    { rule: "Rule A", value: prediction.rule_a },
    { rule: "Rule B", value: prediction.rule_b },
    { rule: "Rule C", value: prediction.rule_c },
    { rule: "Rule D", value: prediction.rule_d },
  ];

  return (
    <ResponsiveContainer width="100%" height={260}>
      <RadarChart data={data} outerRadius="75%">
        <PolarGrid stroke="var(--border-color)" />
        <PolarAngleAxis dataKey="rule" tick={{ fill: "var(--text-light)", fontSize: 12 }} />
        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
        <Radar name="Rule Score" dataKey="value" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.35} isAnimationActive />
        <Tooltip
          formatter={(value: number) => [`${value.toFixed(1)} / 100`, "Score"]}
          contentStyle={{ background: "var(--surface)", border: "1px solid var(--border-color)", borderRadius: 8 }}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
