import { FaExclamationTriangle } from "react-icons/fa";

interface ErrorBannerProps {
  title?: string;
  message: string | null | undefined;
  onRetry?: () => void;
  onDismiss?: () => void;
}

/** Consistent error surface for all failure modes: invalid format, large file, server offline, API timeout, prediction failed, empty response. */
export default function ErrorBanner({ title = "Something went wrong", message, onRetry, onDismiss }: ErrorBannerProps) {
  if (!message) return null;
  return (
    <div className="error-banner d-flex align-items-start gap-3" role="alert">
      <div className="error-banner-icon">
        <FaExclamationTriangle />
      </div>
      <div className="flex-grow-1">
        <strong className="d-block">{title}</strong>
        <p className="mb-0">{message}</p>
      </div>
      <div className="d-flex align-items-center gap-2">
        {onRetry && (
          <button className="btn btn-sm btn-outline-light" onClick={onRetry}>
            Retry
          </button>
        )}
        {onDismiss && (
          <button className="btn-close btn-close-white" onClick={onDismiss} aria-label="Dismiss error" />
        )}
      </div>
    </div>
  );
}
