import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from "recharts";
import type { PredictResponse } from "../../types/api";
import { buildNaturalLanguageExplanation, buildRuleBreakdownSentences } from "../../utils/naturalLanguageExplanation";

interface ExplainabilityPanelProps {
  prediction: PredictResponse;
}

// The 0.35 / 0.30 / 0.20 / 0.15 weights are the real, documented RULE_WEIGHTS
// constant used by the backend's confidence-weighted scoring — used here
// only to compute a *display* contribution breakdown from the already
// -returned rule_a..d scores, not to recompute or override the backend's own
// confidence_weighted_score.
const RULE_WEIGHTS: Record<"rule_a" | "rule_b" | "rule_c" | "rule_d", number> = {
  rule_a: 0.35,
  rule_b: 0.3,
  rule_c: 0.2,
  rule_d: 0.15,
};
const RULE_LABELS: Record<keyof typeof RULE_WEIGHTS, string> = {
  rule_a: "Rule A — Sequential Extension",
  rule_b: "Rule B — Hip Dominance",
  rule_c: "Rule C — Synchronization",
  rule_d: "Rule D — Explosive Movement",
};
const COLORS = ["#22d3ee", "#34d399", "#fbbf24", "#f87171"];

export default function ExplainabilityPanel({ prediction }: ExplainabilityPanelProps) {
  const contributions = (Object.keys(RULE_WEIGHTS) as Array<keyof typeof RULE_WEIGHTS>).map((key) => ({
    label: RULE_LABELS[key],
    value: prediction[key] * RULE_WEIGHTS[key],
  }));
  const total = contributions.reduce((sum, c) => sum + c.value, 0) || 1;
  const chartData = contributions.map((c) => ({ label: c.label, pct: (c.value / total) * 100 }));

  const explanation = buildNaturalLanguageExplanation(prediction);
  const bullets = buildRuleBreakdownSentences(prediction);

  return (
    <div className="explainability-panel">
      <h4>AI Explanation</h4>

      <div className="explainability-section">
        <h6 className="text-uppercase text-accent small fw-bold">Feature Importance (this prediction)</h6>
        <p className="text-muted-2 small">
          How much each rule contributed to this lift's final score, weighted by the documented Rule A–D
          importance weights (0.35 / 0.30 / 0.20 / 0.15) — computed live from this prediction's real data.
        </p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20 }}>
            <XAxis type="number" domain={[0, 100]} tick={{ fill: "var(--text-muted)", fontSize: 11 }} unit="%" />
            <YAxis type="category" dataKey="label" width={180} tick={{ fill: "var(--text-light)", fontSize: 11 }} />
            <Tooltip
              formatter={(value: number) => [`${value.toFixed(1)}%`, "Contribution"]}
              contentStyle={{ background: "var(--surface)", border: "1px solid var(--border-color)", borderRadius: 8 }}
            />
            <Bar dataKey="pct" radius={[0, 6, 6, 0]} isAnimationActive>
              {chartData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="explainability-section">
        <h6 className="text-uppercase text-accent small fw-bold">Natural Language Explanation</h6>
        <p className="nl-explanation">{explanation}</p>
        <ul className="coaching-feedback-list">
          {bullets.map((line, i) => (
            <li key={i}>{line}</li>
          ))}
        </ul>
      </div>

      <div className="explainability-section">
        <div className="d-flex align-items-center gap-2 mb-2">
          <h6 className="text-uppercase text-accent small fw-bold mb-0">Integrated Gradients — Sample Analysis</h6>
          <span className="badge-sample">Sample / Offline</span>
        </div>
        <p className="text-muted-2 small">
          The backend includes a separate Captum Integrated Gradients explainability module
          (<code>src/explainability/explain_prediction.py</code>) not currently exposed by the live
          <code> /predict</code> endpoint. The artifact below is a real, previously-generated sample run
          (not recomputed for this upload) shown here for demonstration.
        </p>
        <img
          src="/sample-explainability/integrated_gradients.png"
          alt="Sample Integrated Gradients attribution heatmap and feature importance bar chart"
          className="explainability-sample-image"
        />
      </div>
    </div>
  );
}
