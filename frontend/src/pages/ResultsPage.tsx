import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { useAnalysis } from "../context/AnalysisContext";
import ScoreBadge from "../components/common/ScoreBadge";
import GlassCard from "../components/common/GlassCard";
import PerformanceGauge from "../components/visualizations/PerformanceGauge";
import ConfidenceGauge from "../components/visualizations/ConfidenceGauge";
import RuleBars from "../components/visualizations/RuleBars";
import RuleRadarChart from "../components/visualizations/RuleRadarChart";
import ROMComparisonChart from "../components/visualizations/ROMComparisonChart";
import VelocityComparisonChart from "../components/visualizations/VelocityComparisonChart";
import MetricCard from "../components/visualizations/MetricCard";
import InfoTooltip from "../components/common/InfoTooltip";
import VideoComparison from "../components/results/VideoComparison";
import ExplainabilityPanel from "../components/results/ExplainabilityPanel";
import DownloadPanel from "../components/results/DownloadPanel";
import { formatDegrees, formatMilliseconds, formatNumber, formatSeconds } from "../utils/formatters";

export default function ResultsPage() {
  const { result, selectedFile, reset } = useAnalysis();
  const navigate = useNavigate();

  useEffect(() => {
    if (!result) navigate("/upload");
  }, [result, navigate]);

  if (!result) return null;

  const p = result.prediction;
  const isGood = p.prediction.toLowerCase() === "good";
  const anthro = p.anthropometric_metrics;

  return (
    <div className="container results-page py-5">
      <motion.div
        className="d-flex justify-content-between align-items-start flex-wrap gap-3 mb-4"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <div>
          <span className="eyebrow">Results Dashboard</span>
          <h2 className="d-flex align-items-center gap-3 flex-wrap">
            <ScoreBadge label={p.prediction} tone={isGood ? "good" : "bad"} size="lg" />
          </h2>
          <p className="text-muted-2 mb-0">
            {result.sourceFileName} — analyzed from {p.n_frames_used} usable frames.
          </p>
        </div>
        <button
          className="btn btn-outline-light"
          onClick={() => {
            reset();
            navigate("/upload");
          }}
        >
          Analyze Another Video
        </button>
      </motion.div>

      {p.low_confidence_flag && (
        <div className="low-confidence-banner mb-4">
          ⚠ This prediction is flagged <strong>low confidence</strong> — treat the classification and scores
          with caution (see Confidence Weighting below).
        </div>
      )}

      <div className="row g-4 mb-4">
        <div className="col-lg-4">
          <GlassCard className="h-100 d-flex flex-column align-items-center justify-content-center text-center">
            <PerformanceGauge score={p.confidence_weighted_score} />
          </GlassCard>
        </div>
        <div className="col-lg-4">
          <GlassCard className="h-100 d-flex flex-column align-items-center justify-content-center text-center">
            <ConfidenceGauge confidence={p.prediction_confidence} lowConfidenceFlag={p.low_confidence_flag} />
          </GlassCard>
        </div>
        <div className="col-lg-4">
          <GlassCard className="h-100">
            <h6 className="text-uppercase text-accent small fw-bold mb-2">Rule Profile</h6>
            <RuleRadarChart prediction={p} />
          </GlassCard>
        </div>
      </div>

      <GlassCard className="mb-4">
        <h6 className="text-uppercase text-accent small fw-bold mb-2 d-flex align-items-center gap-2">
          Rule A–D Breakdown
          <InfoTooltip text="Rule D ('Explosive Movement') is an angular-acceleration proxy approximating rate-of-force-development — not a direct force-plate measurement." />
        </h6>
        <RuleBars prediction={p} />
      </GlassCard>

      <div className="row g-4 mb-4">
        <div className="col-lg-6">
          <GlassCard className="h-100">
            <h6 className="text-uppercase text-accent small fw-bold mb-2">ROM Comparison (Hip vs Knee)</h6>
            <ROMComparisonChart prediction={p} />
          </GlassCard>
        </div>
        <div className="col-lg-6">
          <GlassCard className="h-100">
            <h6 className="text-uppercase text-accent small fw-bold mb-2">Peak Velocity Comparison</h6>
            <VelocityComparisonChart prediction={p} />
          </GlassCard>
        </div>
      </div>

      <GlassCard className="mb-4">
        <h6 className="text-uppercase text-accent small fw-bold mb-3">Biomechanical Metrics</h6>
        <div className="row g-3">
          <div className="col-6 col-md-3">
            <MetricCard label="Hip ROM" value={formatDegrees(p.hip_rom)} tone="accent" />
          </div>
          <div className="col-6 col-md-3">
            <MetricCard label="Knee ROM" value={formatDegrees(p.knee_rom)} tone="accent" />
          </div>
          <div className="col-6 col-md-3">
            <MetricCard
              label="Peak Hip Velocity"
              value={formatNumber(anthro.hip_peak_velocity_normalized)}
              unit="leg-lengths/s"
              tone="accent"
            />
          </div>
          <div className="col-6 col-md-3">
            <MetricCard
              label="Peak Knee Velocity"
              value={formatNumber(anthro.knee_peak_velocity_normalized)}
              unit="leg-lengths/s"
              tone="accent"
            />
          </div>
          <div className="col-6 col-md-3">
            <MetricCard label="Correlation" value={formatNumber(p.correlation, 3)} hint="Hip/knee angle-velocity" tone="neutral" />
          </div>
          <div className="col-6 col-md-3">
            <MetricCard
              label="Sequential Delay"
              value={formatSeconds(p.synchronization)}
              hint="Rule A raw timing delay"
              tone="neutral"
            />
          </div>
          <div className="col-6 col-md-3">
            <MetricCard
              label="Anthropometric Normalization"
              value={formatNumber(anthro.leg_length_reference_px, 1)}
              unit="px ref."
              hint="Leg-length reference used to normalize distances/velocities"
              tone="neutral"
            />
          </div>
          <div className="col-6 col-md-3">
            <MetricCard
              label="Prediction Time"
              value={formatMilliseconds(result.requestDurationMs)}
              hint="Client-measured request duration (not a backend field)"
              tone="neutral"
            />
          </div>
        </div>
      </GlassCard>

      <GlassCard className="mb-4">
        <h6 className="text-uppercase text-accent small fw-bold mb-3">Confidence Weighting</h6>
        <div className="row g-3">
          <div className="col-6 col-md-3">
            <MetricCard label="Rule A Confidence" value={formatNumber(p.rule_confidences.rule_a_confidence * 100, 1)} unit="%" />
          </div>
          <div className="col-6 col-md-3">
            <MetricCard label="Rule B Confidence" value={formatNumber(p.rule_confidences.rule_b_confidence * 100, 1)} unit="%" />
          </div>
          <div className="col-6 col-md-3">
            <MetricCard label="Rule C Confidence" value={formatNumber(p.rule_confidences.rule_c_confidence * 100, 1)} unit="%" />
          </div>
          <div className="col-6 col-md-3">
            <MetricCard label="Rule D Confidence" value={formatNumber(p.rule_confidences.rule_d_confidence * 100, 1)} unit="%" />
          </div>
        </div>
        <p className="text-muted-2 small mt-3 mb-0">
          Overall confidence: {formatNumber(p.rule_confidences.overall_confidence * 100, 1)}% — reflects mean
          YOLO11-pose keypoint confidence blended with the same rule weights used for scoring.
        </p>
      </GlassCard>

      <GlassCard className="mb-4">
        <VideoComparison originalFile={selectedFile} prediction={p} />
      </GlassCard>

      <GlassCard className="mb-4">
        <ExplainabilityPanel prediction={p} />
      </GlassCard>

      <GlassCard>
        <DownloadPanel result={result} />
      </GlassCard>
    </div>
  );
}
