import { BarChart, Bar, XAxis, YAxis, Cell, ResponsiveContainer, Tooltip } from "recharts";
import type { PredictResponse } from "../../types/api";

interface RuleBarsProps {
  prediction: PredictResponse;
}

const RULES: Array<{ key: "rule_a" | "rule_b" | "rule_c" | "rule_d"; label: string }> = [
  { key: "rule_a", label: "Rule A — Sequential Extension" },
  { key: "rule_b", label: "Rule B — Hip Dominance" },
  { key: "rule_c", label: "Rule C — Synchronization" },
  { key: "rule_d", label: "Rule D — Explosive Movement (RFD Approx.)" },
];

function colorFor(score: number): string {
  if (score >= 70) return "var(--good)";
  if (score >= 40) return "var(--warn)";
  return "var(--bad)";
}

/** Rule A-D horizontal bars, real data straight from /predict. */
export default function RuleBars({ prediction }: RuleBarsProps) {
  const data = RULES.map((r) => ({ label: r.label, value: prediction[r.key] }));

  return (
    <div className="rule-bars-chart">
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 20 }}>
          <XAxis type="number" domain={[0, 100]} tick={{ fill: "var(--text-muted)", fontSize: 12 }} />
          <YAxis
            type="category"
            dataKey="label"
            width={190}
            tick={{ fill: "var(--text-light)", fontSize: 12 }}
          />
          <Tooltip
            formatter={(value: number) => [`${value.toFixed(1)} / 100`, "Score"]}
            contentStyle={{ background: "var(--surface)", border: "1px solid var(--border-color)", borderRadius: 8 }}
          />
          <Bar dataKey="value" radius={[0, 6, 6, 0]} isAnimationActive>
            {data.map((d) => (
              <Cell key={d.label} fill={colorFor(d.value)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
