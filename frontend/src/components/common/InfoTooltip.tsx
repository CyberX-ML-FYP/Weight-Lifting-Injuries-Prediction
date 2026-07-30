import { FaInfoCircle } from "react-icons/fa";

interface InfoTooltipProps {
  text: string;
}

/** Lightweight CSS-only hover tooltip (no Bootstrap JS tooltip init required per-instance). */
export default function InfoTooltip({ text }: InfoTooltipProps) {
  return (
    <span className="info-tooltip" tabIndex={0}>
      <FaInfoCircle />
      <span className="info-tooltip-bubble">{text}</span>
    </span>
  );
}
