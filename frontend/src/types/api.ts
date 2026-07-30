// Mirrors src/api/hip_knee_api.py's `PredictResponse` Pydantic model EXACTLY
// (field names, types, and nesting). This file only describes the existing,
// unmodified API contract — it does not change it in any way.

export interface AnthropometricMetrics {
  leg_length_reference_px: number;
  hip_linear_rom_normalized: number;
  knee_linear_rom_normalized: number;
  hip_peak_velocity_normalized: number;
  knee_peak_velocity_normalized: number;
}

export interface RuleConfidences {
  rule_a_confidence: number;
  rule_b_confidence: number;
  rule_c_confidence: number;
  rule_d_confidence: number;
  overall_confidence: number;
}

/** Exact shape of POST /predict's JSON response body. */
export interface PredictResponse {
  prediction: string;
  prediction_confidence: number;
  low_confidence_flag: boolean;
  rule_a: number;
  rule_b: number;
  rule_c: number;
  rule_d: number;
  confidence_weighted_score: number;
  hip_rom: number;
  knee_rom: number;
  hip_peak: number;
  knee_peak: number;
  synchronization: number;
  correlation: number;
  rfd: number;
  anthropometric_metrics: AnthropometricMetrics;
  rule_confidences: RuleConfidences;
  n_frames_used: number;
  annotated_video_path: string | null;
  prediction_json_path: string;
}

/** Exact shape of GET /health's JSON response body. */
export interface HealthResponse {
  status: "ok" | "starting";
  pose_model_loaded: boolean;
}

/** Exact shape of GET /'s JSON response body. */
export interface RootResponse {
  service: string;
  version: string;
  docs: string;
  health: string;
  predict: string;
}

/**
 * Client-side-only wrapper around a real PredictResponse, adding metadata
 * that the frontend itself observed (not returned by the backend, and never
 * presented as if it were): how long the browser's own request took, and the
 * original file it was for. This does not add, rename, or alter any backend
 * field.
 */
export interface AnalysisResult {
  prediction: PredictResponse;
  sourceFileName: string;
  /** Wall-clock duration of the POST /predict request, measured client-side. */
  requestDurationMs: number;
  completedAt: string;
}

export type BackendStatus = "unknown" | "ok" | "starting" | "unreachable";
