import axios, { AxiosError, type AxiosProgressEvent } from "axios";
import type { HealthResponse, PredictResponse } from "../types/api";

// All requests go to relative `/api/*` paths. Vite's dev/preview server
// proxies these to the real, unmodified FastAPI backend on
// http://localhost:8000 (see vite.config.ts) — this file never talks to a
// mock server or fabricates a response of any kind.
const apiClient = axios.create({
  baseURL: "/api",
  timeout: 5 * 60 * 1000, // 5 minutes — pose estimation + LSTM inference on a full video can be slow.
});

export class ApiError extends Error {
  status: number;
  detail?: string;

  constructor(message: string, status: number, detail?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function toApiError(error: unknown): ApiError {
  if (axios.isAxiosError(error)) {
    const axiosErr = error as AxiosError<{ detail?: string }>;
    if (axiosErr.code === "ECONNABORTED") {
      return new ApiError("The request timed out. The backend may be overloaded or unreachable.", 0);
    }
    if (!axiosErr.response) {
      return new ApiError("Backend is unreachable. Is the FastAPI server running on port 8000?", 0);
    }
    const detail = axiosErr.response.data?.detail;
    return new ApiError(
      typeof detail === "string" ? detail : `Request failed (HTTP ${axiosErr.response.status}).`,
      axiosErr.response.status,
      typeof detail === "string" ? detail : undefined,
    );
  }
  return new ApiError("An unexpected error occurred.", 0);
}

/** GET /health — reported exactly as the backend returns it. */
export async function checkHealth(signal?: AbortSignal): Promise<HealthResponse> {
  try {
    const res = await apiClient.get<HealthResponse>("/health", { signal });
    return res.data;
  } catch (error) {
    throw toApiError(error);
  }
}

/**
 * POST /predict — uploads one video (multipart/form-data, field name
 * "video", matching the backend's `UploadFile = File(...)` parameter name
 * exactly) and returns the parsed PredictResponse JSON, unmodified.
 */
export async function predictVideo(
  file: File,
  onProgress?: (loaded: number, total: number) => void,
  signal?: AbortSignal,
): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("video", file);

  try {
    const res = await apiClient.post<PredictResponse>("/predict", formData, {
      signal,
      onUploadProgress: (event: AxiosProgressEvent) => {
        if (event.total && onProgress) {
          onProgress(event.loaded, event.total);
        }
      },
    });
    return res.data;
  } catch (error) {
    throw toApiError(error);
  }
}

export default apiClient;
