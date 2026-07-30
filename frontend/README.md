# Hip & Knee Lift Analytics — Frontend

A production-quality React + TypeScript + Vite frontend for the finalized Hip & Knee
lift-quality analysis backend (`src/api/hip_knee_api.py`). **This frontend does not
modify the backend, the dataset, the model, Rule A–D, the prediction logic, or the
REST API contract in any way** — it only calls the existing endpoints.

## Tech stack

React 18 · Vite · TypeScript · Bootstrap 5 · Axios · React Router 6 · Recharts ·
React Icons · Framer Motion.

## Running locally

1. Start the backend (from the repo root, in the project's Python environment):
   ```
   uvicorn src.api.hip_knee_api:app --host 0.0.0.0 --port 8000
   ```
2. In a second terminal, install and run the frontend:
   ```
   cd frontend
   npm install
   npm run dev
   ```
3. Open the printed local URL (default `http://localhost:5173`).

`npm run build` type-checks (`tsc -b`) and produces a production bundle in
`frontend/dist/`; `npm run preview` serves that bundle locally (proxy behavior is
identical to `dev`).

## How CORS is avoided without touching the backend

The FastAPI app has no `CORSMiddleware` configured, and adding one would violate
the "no backend changes" constraint. Instead, `vite.config.ts` configures Vite's
dev/preview server to proxy any request to `/api/*` through to
`http://localhost:8000/*` (stripping the `/api` prefix) as a same-origin,
server-to-server request. The browser only ever talks to the Vite origin, so no
CORS preflight/rejection ever occurs, and the FastAPI app is completely untouched.

## Application structure

```
frontend/
  src/
    types/api.ts              TypeScript interfaces mirroring PredictResponse exactly
    services/apiClient.ts     Axios instance + GET /health, POST /predict calls
    context/AnalysisContext.tsx  App-wide state machine (idle/uploading/processing/done/error)
    hooks/                    useHealthCheck (polls /health), useCountUp (animated numbers)
    utils/                    validation, formatters, downloads, natural-language explanation
    components/
      layout/                Navbar, Footer, route Layout, ScrollToTop
      common/                ErrorBanner, GlassCard, ProgressBar, SkeletonCard, ScoreBadge, InfoTooltip
      upload/                DropZone, VideoPreview
      visualizations/        PerformanceGauge, ConfidenceGauge, RuleBars, RuleRadarChart,
                              ROMComparisonChart, VelocityComparisonChart, MetricCard
      results/                VideoComparison, ExplainabilityPanel, DownloadPanel
    pages/                   LandingPage, UploadPage, ProcessingPage, ResultsPage, AboutPage
    styles/theme.css          Dark blue/black/white + green/orange/red glass-morphism theme
  public/sample-explainability/  static sample Captum artifacts (labelled "Sample")
```

Routing (`react-router-dom`): `/` (Landing), `/upload`, `/processing`, `/results`,
`/about`. `AnalysisContext` holds the selected file, upload progress, and the real
prediction result across route changes, so `ProcessingPage` and `ResultsPage` can be
navigated to directly while an analysis is in flight or completed.

## Known limitations (disclosed intentionally)

- **Annotated video is path-only.** `PredictResponse.annotated_video_path` is a
  server-side filesystem path, not an HTTP URL — the backend has no static-file
  or streaming route for it. The video comparison view displays this path as text
  with an explanation rather than faking a working video player.
- **Live, per-request Captum explainability does not exist in the API.**
  `src/explainability/explain_prediction.py` is a separate, offline CLI script
  not wired into `/predict`. The AI Explanation panel instead shows: (a) a
  genuinely live "feature importance" chart computed only from the real
  `rule_a`–`rule_d` values and the documented `RULE_WEIGHTS` (0.35/0.30/0.20/0.15),
  and (b) a clearly `Sample`-labelled bundle of a previously-generated Captum
  Integrated Gradients run, bundled as static files under
  `frontend/public/sample-explainability/`. Neither is presented as live,
  per-upload Captum analysis.
- **"Prediction Time" is client-measured**, not a backend field — the API's
  `PredictResponse` has no timing field, so this is the wall-clock duration the
  browser observed for the `POST /predict` request, labelled as such.
- **Natural-language explanation and coaching feedback are a local heuristic**,
  not the backend's scoring logic. They only phrase the real returned Rule A–D /
  score / confidence values using simple display bands (≥70 / 40–69 / <40); they
  never recompute or override the backend's actual adaptive-threshold
  classification.
- Footer academic details (university/authors/supervisor) are placeholders —
  fill these in with real values before submission/demo.
