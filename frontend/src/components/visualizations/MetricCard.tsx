interface MetricCardProps {
  label: string;
  value: string;
  unit?: string;
  hint?: string;
  tone?: "accent" | "good" | "warn" | "bad" | "neutral";
}

export default function MetricCard({ label, value, unit, hint, tone = "neutral" }: MetricCardProps) {
  return (
    <div className={`metric-card metric-card-${tone}`}>
      <span className="metric-card-label">{label}</span>
      <span className="metric-card-value">
        {value}
        {unit && <span className="metric-card-unit">{unit}</span>}
      </span>
      {hint && <span className="metric-card-hint">{hint}</span>}
    </div>
  );
}
