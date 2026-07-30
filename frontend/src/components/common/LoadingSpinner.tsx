interface LoadingSpinnerProps {
  size?: number;
}

export default function LoadingSpinner({ size = 64 }: LoadingSpinnerProps) {
  return (
    <div className="loading-spinner" style={{ width: size, height: size }} role="status" aria-label="Loading">
      <div className="loading-spinner-ring" />
    </div>
  );
}
