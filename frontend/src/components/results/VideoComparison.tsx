import { useEffect, useMemo, useRef } from "react";
import { FaExpand } from "react-icons/fa";
import type { PredictResponse } from "../../types/api";

interface VideoComparisonProps {
  originalFile: File | null;
  prediction: PredictResponse;
}

/**
 * Original vs. Annotated video, side-by-side, with playback controls and a
 * fullscreen toggle.
 *
 * Honest limitation: `PredictResponse.annotated_video_path` is a server-side
 * filesystem path — the unmodified backend has no static-file/streaming
 * route serving it over HTTP. Per the "no backend/API changes" constraint,
 * this component does not fabricate a working player for a file it cannot
 * fetch; it shows the real returned path transparently with an explanation
 * instead of a broken/fake video.
 */
export default function VideoComparison({ originalFile, prediction }: VideoComparisonProps) {
  const originalRef = useRef<HTMLDivElement | null>(null);

  function toggleFullscreen(ref: React.RefObject<HTMLDivElement>) {
    const el = ref.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      el.requestFullscreen?.();
    }
  }

  const originalUrl = useMemo(() => (originalFile ? URL.createObjectURL(originalFile) : null), [originalFile]);

  useEffect(() => {
    return () => {
      if (originalUrl) URL.revokeObjectURL(originalUrl);
    };
  }, [originalUrl]);

  return (
    <div className="video-comparison">
      <h4 className="mb-3">Video Viewer</h4>
      <div className="row g-4">
        <div className="col-md-6">
          <div className="video-pane" ref={originalRef}>
            <div className="video-pane-header">
              <span>Original Upload</span>
              <button className="btn btn-sm btn-outline-light" onClick={() => toggleFullscreen(originalRef)}>
                <FaExpand />
              </button>
            </div>
            {originalUrl ? (
              <video src={originalUrl} controls className="video-el" />
            ) : (
              <p className="text-muted-2 p-4">Original video is not available in this session.</p>
            )}
          </div>
        </div>

        <div className="col-md-6">
          <div className="video-pane">
            <div className="video-pane-header">
              <span>Annotated Output (pose overlay)</span>
            </div>
            {prediction.annotated_video_path ? (
              <div className="video-unavailable p-4 text-center">
                <p className="mb-2">
                  The backend generated an annotated video and reported its server-side path:
                </p>
                <code className="video-path d-block mb-2">{prediction.annotated_video_path}</code>
                <p className="text-muted-2 small mb-0">
                  The current API only returns a filesystem path, not a streamable URL, so it can't be played
                  here without adding a static-file/video-serving route to the backend — out of scope for
                  this task. Retrieve it directly from the server's{" "}
                  <code>reports/figures/hip_knee_analysis/annotated_videos/</code> directory.
                </p>
              </div>
            ) : (
              <p className="text-muted-2 p-4">
                No annotated video was generated for this prediction (rendering may have failed while the
                JSON report still succeeded).
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
