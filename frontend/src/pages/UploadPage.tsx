import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import DropZone from "../components/upload/DropZone";
import VideoPreview from "../components/upload/VideoPreview";
import ProgressBar from "../components/common/ProgressBar";
import ErrorBanner from "../components/common/ErrorBanner";
import GlassCard from "../components/common/GlassCard";
import { validateVideoFile } from "../utils/validation";
import { formatBytes } from "../utils/validation";
import { useAnalysis } from "../context/AnalysisContext";
import { useHealthCheck } from "../hooks/useHealthCheck";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const { status, uploadProgress, startAnalysis, cancelAnalysis } = useAnalysis();
  const backendStatus = useHealthCheck();
  const navigate = useNavigate();

  const isUploading = status === "uploading";

  function handleFileSelected(candidate: File) {
    const result = validateVideoFile(candidate);
    if (!result.valid) {
      setValidationError(result.reason ?? "Invalid file.");
      setFile(null);
      return;
    }
    setValidationError(null);
    setFile(candidate);
  }

  async function handleAnalyze() {
    if (!file) return;
    navigate("/processing");
    await startAnalysis(file);
  }

  return (
    <div className="container upload-page py-5">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <h2>Upload a Lift Video</h2>
        <p className="text-muted-2">
          Select a single side-view lift clip. The video is analyzed once you click Analyze — nothing is sent
          to the server until then.
        </p>
      </motion.div>

      <ErrorBanner title="Invalid file" message={validationError} onDismiss={() => setValidationError(null)} />

      {backendStatus === "unreachable" && (
        <ErrorBanner
          title="Backend unavailable"
          message="The FastAPI backend could not be reached. Start it (uvicorn src.api.hip_knee_api:app) before analyzing a video."
        />
      )}

      <div className="row g-4 my-2">
        <div className="col-lg-6">
          <GlassCard>
            <DropZone onFileSelected={handleFileSelected} />
            {file && !isUploading && (
              <div className="mt-3 d-flex justify-content-between align-items-center">
                <span className="text-muted-2 small">{formatBytes(file.size)} selected</span>
                <button className="btn btn-sm btn-outline-light" onClick={() => setFile(null)}>
                  Remove file
                </button>
              </div>
            )}
          </GlassCard>
        </div>
        <div className="col-lg-6">
          {file ? (
            <GlassCard>
              <VideoPreview file={file} label="Preview" />
            </GlassCard>
          ) : (
            <GlassCard className="d-flex align-items-center justify-content-center text-muted-2" style={{ minHeight: 200 }}>
              No video selected yet.
            </GlassCard>
          )}
        </div>
      </div>

      {isUploading && (
        <GlassCard className="my-4">
          <ProgressBar
            label="Uploading…"
            sublabel={`${Math.round((uploadProgress.loaded / Math.max(uploadProgress.total, 1)) * 100)}%`}
            value={uploadProgress.loaded}
            max={uploadProgress.total || 1}
          />
          <button className="btn btn-sm btn-outline-danger mt-2" onClick={cancelAnalysis}>
            Cancel Upload
          </button>
        </GlassCard>
      )}

      <div className="d-flex gap-3">
        <button
          className="btn btn-accent btn-lg"
          disabled={!file || isUploading || backendStatus === "unreachable"}
          onClick={handleAnalyze}
        >
          Analyze Lift →
        </button>
      </div>
    </div>
  );
}
