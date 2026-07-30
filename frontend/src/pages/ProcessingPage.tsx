import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FaCheckCircle, FaCircleNotch } from "react-icons/fa";
import LoadingSpinner from "../components/common/LoadingSpinner";
import ProgressBar from "../components/common/ProgressBar";
import ErrorBanner from "../components/common/ErrorBanner";
import GlassCard from "../components/common/GlassCard";
import { useAnalysis } from "../context/AnalysisContext";

// These stages describe the REAL pipeline steps that
// src/models/hip_knee_predict.py::predict_video runs (pose extraction,
// feature/biomechanics computation, LSTM classification, explainability
// hooks, report generation), in their real execution order. Because
// POST /predict is a single blocking HTTP call (the API contract has no
// server-sent progress events), this timeline advances on an estimated
// timer as an illustrative indicator — it is not a literal real-time
// per-stage signal from the backend.
const STAGES = [
  "Pose Extraction (YOLO11)",
  "Feature Extraction",
  "Biomechanical Analysis (Rule A–D)",
  "LSTM Prediction",
  "Explainability",
  "Generating Report",
];

const ESTIMATED_TOTAL_MS = 45000; // rough illustrative estimate for a typical clip

export default function ProcessingPage() {
  const { status, uploadProgress, error, result } = useAnalysis();
  const navigate = useNavigate();
  const [serverElapsedMs, setServerElapsedMs] = useState(0);

  useEffect(() => {
    if (status === "done" && result) {
      navigate("/results");
    }
  }, [status, result, navigate]);

  useEffect(() => {
    if (status === "idle") {
      navigate("/upload");
    }
  }, [status, navigate]);

  useEffect(() => {
    if (status !== "processing") return undefined;
    const start = performance.now();
    const interval = setInterval(() => setServerElapsedMs(performance.now() - start), 250);
    return () => clearInterval(interval);
  }, [status]);

  const uploadPct = uploadProgress.total > 0 ? Math.round((uploadProgress.loaded / uploadProgress.total) * 100) : 0;
  const serverPct = Math.min(96, Math.round((serverElapsedMs / ESTIMATED_TOTAL_MS) * 100));
  const overallPct = status === "uploading" ? Math.round(uploadPct * 0.15) : 15 + Math.round(serverPct * 0.85);
  const activeStageIndex = Math.min(STAGES.length - 1, Math.floor((serverPct / 100) * STAGES.length));
  const etaSeconds = Math.max(0, Math.round((ESTIMATED_TOTAL_MS - serverElapsedMs) / 1000));

  return (
    <div className="container processing-page py-5 d-flex justify-content-center">
      <GlassCard className="processing-card text-center">
        {status === "error" ? (
          <>
            <ErrorBanner title="Prediction failed" message={error} onRetry={() => navigate("/upload")} />
          </>
        ) : (
          <>
            <LoadingSpinner />
            <h2 className="mt-3">Analyzing Your Lift</h2>
            <p className="text-muted-2">
              {status === "uploading" ? `Uploading video… ${uploadPct}%` : "Running the full analysis pipeline on the server…"}
            </p>

            <ProgressBar value={overallPct} max={100} tone="accent" />
            <div className="d-flex justify-content-between text-muted-2 small mt-1">
              <span>{overallPct}% complete</span>
              {status === "processing" && <span>Est. {etaSeconds}s remaining</span>}
            </div>

            <ul className="processing-timeline mt-4 text-start">
              {STAGES.map((stage, i) => {
                const isDone = status === "processing" ? i < activeStageIndex : false;
                const isActive = status === "processing" && i === activeStageIndex;
                return (
                  <li key={stage} className={`timeline-item ${isDone ? "timeline-done" : ""} ${isActive ? "timeline-active" : ""}`}>
                    <span className="timeline-icon">
                      {isDone ? <FaCheckCircle /> : isActive ? <FaCircleNotch className="spin-icon" /> : <span className="timeline-dot" />}
                    </span>
                    <span>{stage}</span>
                  </li>
                );
              })}
            </ul>

            <p className="processing-note text-muted-2 small">
              Larger / higher-resolution videos can take longer — pose estimation is typically the slowest step.
              Progress above is an illustrative estimate, since the API returns one combined result at the end.
            </p>
          </>
        )}
      </GlassCard>
    </div>
  );
}
