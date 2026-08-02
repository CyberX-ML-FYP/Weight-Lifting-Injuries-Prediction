interface ProgressBarProps {
  value: number;
  max?: number;
  label?: string;
  sublabel?: string;
  tone?: "accent" | "good" | "warn" | "bad";
}

export default function ProgressBar({ value, max = 100, label, sublabel, tone = "accent" }: ProgressBarProps) {
  const pct = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;
  return (
    <div className="progress-bar-wrap">
      {(label || sublabel) && (
        <div className="progress-bar-labels">
          {label && <span>{label}</span>}
          {sublabel && <span className="progress-bar-sublabel">{sublabel}</span>}
        </div>
      )}
      <div className="progress-bar-track">
        <div className={`progress-bar-fill progress-bar-${tone}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
