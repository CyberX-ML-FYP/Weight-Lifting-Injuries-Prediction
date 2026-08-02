interface ScoreBadgeProps {
  label: string;
  tone?: "good" | "warn" | "bad" | "neutral";
  size?: "sm" | "md" | "lg";
}

const TONE_CLASS: Record<string, string> = {
  good: "badge-good",
  warn: "badge-warn",
  bad: "badge-bad",
  neutral: "badge-neutral",
};

export default function ScoreBadge({ label, tone = "neutral", size = "md" }: ScoreBadgeProps) {
  return <span className={`score-badge score-badge-${size} ${TONE_CLASS[tone]}`}>{label}</span>;
}
