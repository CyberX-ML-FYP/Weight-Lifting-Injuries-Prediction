# Hip & Knee Prediction API — Documentation

**Module:** `src/api/hip_knee_api.py`
**Scope:** Hip & Knee (lower-body) analysis pipeline only. Does not modify,
import from, or duplicate logic from `module3_arm_analysis`, the frontend,
or any other teammate module. Does not retrain or alter the trained model.

The API is a thin HTTP wrapper around the already-validated prediction
pipeline (`src.models.hip_knee_predict.predict_video`, see
[reports/backend_validation.md](backend_validation.md)). All analysis logic
(frame extraction, YOLO11 pose, Rule A-D biomechanics, anthropometric
normalization, confidence weighting, pose smoothing, LSTM classification,
annotated-video rendering) is reused as-is — nothing is duplicated.

---

## 1. Running the API

```powershell
# Install dependencies (already added to requirements.txt)
pip install fastapi "uvicorn[standard]" python-multipart

# Start the server
python -m uvicorn src.api.hip_knee_api:app --host 0.0.0.0 --port 8000
```

Interactive Swagger UI is auto-generated at `http://localhost:8000/docs`
(FastAPI/OpenAPI default).

The YOLO11-pose model is loaded **once** at server startup (not per
request) via a FastAPI `lifespan` handler, so the first request after
startup is not penalized by model-loading time. `/health` reports whether
this has finished.

---

## 2. Endpoints

### `GET /`
Basic service metadata (name, version, links to `/docs` and `/health`).

### `GET /health`
Liveness/readiness check.

```json
{ "status": "ok", "pose_model_loaded": true }
```

`status` is `"starting"` until the YOLO pose model has finished loading.

### `POST /predict`

Runs the full Hip & Knee prediction pipeline on one uploaded video.

- **Content type:** `multipart/form-data`
- **Field:** `video` — the lift video file (`.mp4` or `.mov`, case-insensitive)
- **Max size:** 500 MB (returns `413` if exceeded)

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "video=@13bad.mp4;type=video/mp4"
```

**Example (Python):**
```python
import requests

with open("13bad.mp4", "rb") as f:
    resp = requests.post("http://localhost:8000/predict", files={"video": f})
resp.raise_for_status()
report = resp.json()
```

#### Success response — `200 OK`

```json
{
  "prediction": "Good",
  "prediction_confidence": 0.5222,
  "low_confidence_flag": true,
  "rule_a": 50.0,
  "rule_b": 9.06,
  "rule_c": 99.90,
  "rule_d": 9.19,
  "confidence_weighted_score": 70.70,
  "hip_rom": 43.98,
  "knee_rom": 46.72,
  "hip_peak": 179.37,
  "knee_peak": 181.76,
  "synchronization": 0.0,
  "correlation": 0.9964,
  "rfd": 0.9953,
  "anthropometric_metrics": {
    "leg_length_reference_px": 601.82,
    "hip_linear_rom_normalized": 0.0646,
    "knee_linear_rom_normalized": 0.4248,
    "hip_peak_velocity_normalized": 0.8121,
    "knee_peak_velocity_normalized": 6.3027
  },
  "rule_confidences": {
    "rule_a_confidence": 0.9939,
    "rule_b_confidence": 0.9939,
    "rule_c_confidence": 0.9779,
    "rule_d_confidence": 0.9939,
    "overall_confidence": 0.9907
  },
  "n_frames_used": 20,
  "annotated_video_path": "D:/.../reports/annotated_videos/<request_id>_annotated.mp4",
  "prediction_json_path": "D:/.../reports/api_predictions/<request_id>.json"
}
```

(Response captured from a real run against `data/raw/hip_knee/Side view/13bad.MOV`.)

#### Response field reference

| Field | Type | Source | Description |
|---|---|---|---|
| `prediction` | string | LSTM | Predicted class (`Good`/`Average`/`Poor`) |
| `prediction_confidence` | float | LSTM | Softmax confidence for the predicted class (0-1) |
| `low_confidence_flag` | bool | Confidence weighting | `true` if model or rule confidence is below the low-confidence threshold |
| `rule_a` / `rule_b` / `rule_c` / `rule_d` | float | Rule A-D biomechanics | 0-100 normalized rule scores |
| `confidence_weighted_score` | float | Confidence weighting | Rule A-D score blended by per-rule confidence, 0-100 |
| `hip_rom` / `knee_rom` | float | Biomechanics | Range of motion in degrees |
| `hip_peak` / `knee_peak` | float | Biomechanics | Peak joint angle in degrees |
| `synchronization` | float | Rule C | Hip/knee synchronization delay in seconds (see limitation in §5) |
| `correlation` | float | Rule C | Hip/knee angle-velocity correlation |
| `rfd` | float | Rule D | Rate of force development |
| `anthropometric_metrics.*` | float | Anthropometric normalization | Leg-length reference + normalized linear ROM/velocity |
| `rule_confidences.*` | float | Confidence weighting | Per-rule + overall confidence (extra field beyond the minimum request, included for transparency) |
| `n_frames_used` | int | Pipeline | Number of sampled frames that passed the pose-quality gate |
| `annotated_video_path` | string \| null | Pipeline output | Path to the rendered overlay video, or `null` if generation failed/was skipped (isolated failure — does not fail the whole request, see [backend_validation.md](backend_validation.md) fix #4) |
| `prediction_json_path` | string | Pipeline output | Path to the full structured prediction JSON written to disk |

#### Error responses

| Status | Cause | Example detail |
|---|---|---|
| `422 Unprocessable Entity` | Unsupported file extension | `"Unsupported file extension '.txt'. Allowed: .mp4, .mov"` |
| `422 Unprocessable Entity` | Empty (0-byte) upload | `"Uploaded video file is empty (0 bytes)."` |
| `422 Unprocessable Entity` | Corrupted/unreadable video (ffmpeg decode failure) | `"ffmpeg failed to decode '<file>' — the file may be corrupted or in an unsupported/unreadable format. ffmpeg stderr (tail): ..."` |
| `422 Unprocessable Entity` | No usable frames (all rejected by the pose-quality gate) | `"No usable frames extracted from <path>"` |
| `413 Payload Too Large` | Upload exceeds 500 MB | `"Uploaded video exceeds the 500 MB limit."` |
| `500 Internal Server Error` | Unexpected/unclassified failure | `"Internal error during prediction."` (full traceback logged server-side only) |

All error responses were verified against real requests (unsupported
extension, 0-byte file, ffmpeg-unreadable file) during this task — see §4.

---

## 3. Design notes

- **No duplicated logic.** The endpoint calls
  `src.models.hip_knee_predict.predict_video()` directly; the response
  model only reshapes the returned `PredictionReport` dataclass.
- **Per-request isolation.** Each upload gets a UUID-based `request_id`
  used for its temp upload path, frame-cache directory, JSON output path,
  and annotated-video filename — concurrent requests (or requests with the
  same original filename) never collide or overwrite each other's output.
- **Cleanup.** The temporary uploaded video and its per-request frame
  cache are always deleted after the request (success or failure); the
  annotated video and prediction JSON are kept in `reports/annotated_videos/`
  and `reports/api_predictions/` respectively as the durable outputs.
- **Upload safety.** Uploads are streamed to disk in 1 MB chunks with a
  500 MB hard cap (returns `413` if exceeded) instead of buffering the
  whole file in memory — mitigates unbounded resource consumption from a
  large/malicious upload (OWASP CWE-400). The original filename is never
  used to build a filesystem path (a UUID is used instead), which
  eliminates path-traversal risk from a crafted filename.
- **Concurrency.** FastAPI runs synchronous endpoints in a thread pool, so
  multiple uploads can arrive concurrently, but the actual YOLO/LSTM
  inference call is serialized behind a lock (`_inference_lock`) since
  Ultralytics/PyTorch model objects are not guaranteed thread-safe for
  concurrent forward passes from multiple threads. This means requests are
  queued rather than run in parallel — acceptable for this scope, noted as
  a scaling limitation below.
- **Error-handling boundary.** Mirrors the CLI's `main()` design (see
  [backend_validation.md](backend_validation.md) fix #3): known/expected
  failure modes (`FileNotFoundError`, `ValueError`, `RuntimeError`) map to a
  clean `422` with the underlying message; anything unexpected maps to a
  generic `500` with the real error only in the server log (avoids leaking
  internal details to API clients).

---

## 4. Verification performed

- Started the API locally (`uvicorn src.api.hip_knee_api:app`), confirmed
  `/health` reports `pose_model_loaded: true` after startup.
- `POST /predict` with a real video (`data/raw/hip_knee/Side view/13bad.MOV`)
  → `200 OK` with a fully-populated response; confirmed both
  `annotated_video_path` and `prediction_json_path` exist on disk.
- `POST /predict` with an unsupported/corrupted file (text content renamed
  to `.mp4`) → `422` with a clear ffmpeg-decode error message.
- `POST /predict` with a 0-byte file → `422` with a clear empty-file message.
- `POST /predict` with a `.txt` file → `422` rejected by extension check
  before any processing.

---

## 5. Known limitations

- **Single-worker inference.** The `_inference_lock` means only one video
  is analyzed at a time regardless of how many requests arrive
  concurrently; under load, requests queue rather than run in parallel.
  Scaling further (e.g. multiple worker processes, each with its own
  loaded model) would require moving inference to a task queue — out of
  scope for this task.
- **Synchronization-delay units.** Inherited from the underlying pipeline
  (see [backend_validation.md](backend_validation.md) §4.1): the
  `synchronization` field's units assume the fixed 20-samples convention,
  not true video-time seconds for atypical fps/duration videos.
- **No authentication/rate-limiting.** This API has no auth layer or
  request throttling — appropriate for an internal/trusted-network
  deployment as scoped by this task, but should be added (e.g. an API key
  or reverse-proxy auth) before exposing it on a public network.
- **CORS is not configured.** Consumers embedding this API behind a
  browser-based frontend will need to add their own CORS middleware/config
  for their specific origin; a permissive default was deliberately not
  added here to avoid an insecure default.
