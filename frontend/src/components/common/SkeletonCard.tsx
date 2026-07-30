interface SkeletonCardProps {
  height?: number;
  className?: string;
}

/** Animated skeleton placeholder shown while data/visualizations are loading. */
export default function SkeletonCard({ height = 120, className = "" }: SkeletonCardProps) {
  return <div className={`skeleton-card ${className}`} style={{ height }} aria-hidden="true" />;
}
