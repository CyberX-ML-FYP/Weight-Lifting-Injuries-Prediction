import { useEffect, useMemo } from "react";
import { formatBytes } from "../../utils/validation";

interface VideoPreviewProps {
  file: File;
  label?: string;
}

/**
 * Plays the file the user actually selected, straight from the browser's
 * local Blob URL — no upload round-trip needed just to preview it. Reused
 * on the Results page as the "Original" side of the video comparison, since
 * the backend deletes the temporary uploaded file after each prediction
 * request and does not expose an endpoint to re-fetch it.
 */
export default function VideoPreview({ file, label = "Selected video" }: VideoPreviewProps) {
  const objectUrl = useMemo(() => URL.createObjectURL(file), [file]);

  useEffect(() => {
    return () => URL.revokeObjectURL(objectUrl);
  }, [objectUrl]);

  return (
    <div className="video-preview">
      <video src={objectUrl} controls preload="metadata" className="video-el" />
      <div className="video-preview-meta">
        <span className="video-preview-label">{label}</span>
        <span>{file.name}</span>
        <span>{formatBytes(file.size)}</span>
      </div>
    </div>
  );
}
