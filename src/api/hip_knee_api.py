"""
Hip & Knee Analysis — Production REST API.

Exposes the existing, already-validated Hip & Knee prediction pipeline
(``src.models.hip_knee_predict.predict_video``) over HTTP so external
clients (a frontend, another service, etc.) can submit one lift video and
receive a structured prediction report — without reimplementing any of the
pipeline logic (frame extraction, pose estimation, Rule A-D biomechanics,
anthropometric normalization, confidence weighting, pose smoothing, LSTM
classification, or annotated-video rendering all continue to live in their
existing modules and are simply called from here).

Endpoints:
    POST /predict  — upload a video, run the full pipeline, return JSON.
    GET  /health   — liveness/readiness check (reports whether the
                     pose model has finished loading).
    GET  /         — basic API metadata.

Scope: this module belongs to the Hip & Knee (lower-body) analysis
pipeline only. It does NOT modify, import from, or duplicate logic from
``module3_arm_analysis``, the frontend, or any other teammate module, and
it does NOT retrain or alter the trained model in any way.
"""
from __future__ import annotations

import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.features.hip_knee_config import INTERIM_DIR, VIDEO_EXTENSIONS
from src.features.hip_knee_logger import get_logger
from src.models.hip_knee_predict import (
    DEFAULT_ANNOTATED_VIDEO_DIR,
    PredictionReport,
    REPORTS_DIR,
    load_pose_model,
    predict_video,
)

logger = get_logger(__name__)

# ── API-specific scratch/output locations ─────────────────────────────────────
# Kept separate from the CLI's default paths (``data/interim/hip_knee/predict``,
# ``reports/prediction.json``) so concurrent API requests never collide with
# each other or with a manually-run CLI prediction.
API_UPLOAD_DIR: Path = INTERIM_DIR / "api_uploads"
API_FRAMES_ROOT: Path = INTERIM_DIR / "api_predict"
API_JSON_DIR: Path = REPORTS_DIR / "api_predictions"

# 500 MB safety cap on uploaded video size — prevents unbounded disk/memory
# consumption from a malicious or accidental oversized upload (OWASP: CWE-400
# Uncontrolled Resource Consumption).
MAX_UPLOAD_BYTES: int = 500 * 1024 * 1024
_UPLOAD_CHUNK_BYTES: int = 1024 * 1024

# Ultralytics YOLO / the LSTM classifier are not guaranteed thread-safe for
# concurrent forward passes. FastAPI runs sync `def` endpoints in a thread
# pool, so multiple uploads could arrive concurrently — this lock serializes
# the actual inference call (requests are queued, not parallelized) to avoid
# undefined behaviour rather than building a full task-queue for this scope.
_inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the (comparatively expensive) YOLO11-pose model once at startup."""
    logger.info("Loading YOLO11-pose model at API startup...")
    app.state.pose_model = load_pose_model()
    logger.info("YOLO11-pose model loaded — API ready to accept requests.")
    yield
    app.state.pose_model = None


app = FastAPI(
    title="Hip & Knee Lift-Quality Prediction API",
    description=(
        "Upload a lift video and receive the full Hip & Knee analysis: LSTM "
        "prediction, Rule A-D biomechanics scores, anthropometric "
        "normalization, confidence weighting, and an annotated output video."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Response schema ────────────────────────────────────────────────────────────
class AnthropometricMetrics(BaseModel):
    leg_length_reference_px: float
    hip_linear_rom_normalized: float
    knee_linear_rom_normalized: float
    hip_peak_velocity_normalized: float
    knee_peak_velocity_normalized: float


class RuleConfidences(BaseModel):
    """Per-rule confidence scores backing ``confidence_weighted_score`` (see
    reports/confidence_weighted_rules.md). Included for full transparency in
    addition to the minimum required response fields."""

    rule_a_confidence: float
    rule_b_confidence: float
    rule_c_confidence: float
    rule_d_confidence: float
    overall_confidence: float


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Predicted lift-quality class, e.g. 'Good' or 'Poor'.")
    prediction_confidence: float = Field(..., description="LSTM softmax confidence for the predicted class (0-1).")
    low_confidence_flag: bool = Field(..., description="True if either the model or the rule confidences are below the low-confidence threshold.")
    rule_a: float = Field(..., description="Rule A (sequential extension) score, 0-100.")
    rule_b: float = Field(..., description="Rule B (hip dominance) score, 0-100.")
    rule_c: float = Field(..., description="Rule C (hip/knee synchronization) score, 0-100.")
    rule_d: float = Field(..., description="Rule D (rate of force development) score, 0-100.")
    confidence_weighted_score: float = Field(..., description="Rule A-D score blended by per-rule confidence, 0-100.")
    hip_rom: float = Field(..., description="Hip range of motion, degrees.")
    knee_rom: float = Field(..., description="Knee range of motion, degrees.")
    hip_peak: float = Field(..., description="Peak hip angle, degrees.")
    knee_peak: float = Field(..., description="Peak knee angle, degrees.")
    synchronization: float = Field(..., description="Hip/knee synchronization delay, seconds (see report limitations re: fps convention).")
    correlation: float = Field(..., description="Hip/knee angle-velocity correlation coefficient.")
    rfd: float = Field(..., description="Rate of force development metric.")
    anthropometric_metrics: AnthropometricMetrics
    rule_confidences: RuleConfidences
    n_frames_used: int = Field(..., description="Number of sampled frames that passed the pose-quality gate.")
    annotated_video_path: Optional[str] = Field(None, description="Path to the annotated output video, or null if video generation failed/was skipped.")
    prediction_json_path: str = Field(..., description="Path to the full structured prediction JSON on disk.")


def _report_to_response(report: PredictionReport, json_path: Path, video_path: Path) -> PredictResponse:
    """Map the existing ``PredictionReport`` dataclass onto the API's response
    schema — no analysis logic lives here, this only reshapes already-computed
    fields."""
    return PredictResponse(
        prediction=report.predicted_class,
        prediction_confidence=report.prediction_confidence,
        low_confidence_flag=report.is_low_confidence,
        rule_a=report.rule_a_score,
        rule_b=report.rule_b_score,
        rule_c=report.rule_c_score,
        rule_d=report.rule_d_score,
        confidence_weighted_score=report.confidence_weighted_final_score,
        hip_rom=report.hip_rom,
        knee_rom=report.knee_rom,
        hip_peak=report.hip_peak,
        knee_peak=report.knee_peak,
        synchronization=report.synchronization_delay,
        correlation=report.correlation,
        rfd=report.rate_of_force_development,
        anthropometric_metrics=AnthropometricMetrics(
            leg_length_reference_px=report.leg_length_reference_px,
            hip_linear_rom_normalized=report.hip_linear_rom_normalized,
            knee_linear_rom_normalized=report.knee_linear_rom_normalized,
            hip_peak_velocity_normalized=report.hip_peak_velocity_normalized,
            knee_peak_velocity_normalized=report.knee_peak_velocity_normalized,
        ),
        rule_confidences=RuleConfidences(
            rule_a_confidence=report.rule_a_confidence,
            rule_b_confidence=report.rule_b_confidence,
            rule_c_confidence=report.rule_c_confidence,
            rule_d_confidence=report.rule_d_confidence,
            overall_confidence=report.overall_confidence,
        ),
        n_frames_used=report.n_frames_used,
        annotated_video_path=video_path.as_posix() if video_path.exists() else None,
        prediction_json_path=json_path.as_posix(),
    )


def _save_upload(upload: UploadFile, destination: Path) -> None:
    """
    Stream the uploaded file to disk in chunks, enforcing ``MAX_UPLOAD_BYTES``
    without reading the whole file into memory first.
    """
    total = 0
    try:
        with destination.open("wb") as out_file:
            while True:
                chunk = upload.file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Uploaded video exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                    )
                out_file.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.") from exc
    finally:
        upload.file.close()

    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Uploaded video file is empty (0 bytes).")


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "Hip & Knee Lift-Quality Prediction API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict (multipart/form-data, field name 'video')",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    pose_model_ready = getattr(app.state, "pose_model", None) is not None
    return {"status": "ok" if pose_model_ready else "starting", "pose_model_loaded": pose_model_ready}


@app.post("/predict", response_model=PredictResponse)
def predict(video: UploadFile = File(..., description="Lift video file (.mp4 or .mov).")) -> PredictResponse:
    """
    Run the full Hip & Knee prediction pipeline on an uploaded video.

    Reuses ``src.models.hip_knee_predict.predict_video`` end-to-end (frame
    extraction, YOLO11 pose, Rule A-D biomechanics, anthropometric
    normalization, confidence weighting, pose smoothing, LSTM prediction,
    JSON + annotated-video output) — no pipeline logic is duplicated here.
    """
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file extension '{suffix or '(none)'}'. Allowed: {', '.join(VIDEO_EXTENSIONS)}",
        )

    request_id = uuid.uuid4().hex
    API_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    API_JSON_DIR.mkdir(parents=True, exist_ok=True)

    temp_video_path = API_UPLOAD_DIR / f"{request_id}{suffix}"
    frames_root = API_FRAMES_ROOT / request_id
    output_json_path = API_JSON_DIR / f"{request_id}.json"
    output_video_path = DEFAULT_ANNOTATED_VIDEO_DIR / f"{request_id}_annotated.mp4"

    _save_upload(video, temp_video_path)

    try:
        with _inference_lock:
            report = predict_video(
                video_path=temp_video_path,
                frames_root=frames_root,
                pose_model=app.state.pose_model,
                output_json_path=output_json_path,
                output_video_path=output_video_path,
                save_video=True,
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        # Known/expected failure modes (empty/corrupted/unsupported video, no
        # usable frames, etc.) — same exceptions the CLI's main() catches.
        logger.exception("Prediction failed for uploaded video '%s' (request_id=%s)", video.filename, request_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - deliberate top-level safety net for a public API
        logger.exception("Unexpected error during prediction (request_id=%s)", request_id)
        raise HTTPException(status_code=500, detail="Internal error during prediction.") from exc
    finally:
        temp_video_path.unlink(missing_ok=True)
        shutil.rmtree(frames_root, ignore_errors=True)

    return _report_to_response(report, output_json_path, output_video_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.hip_knee_api:app", host="0.0.0.0", port=8000, reload=False)
