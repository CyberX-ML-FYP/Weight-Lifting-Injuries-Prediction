import React, { createContext, useCallback, useContext, useRef, useState } from "react";
import { predictVideo, ApiError } from "../services/apiClient";
import type { AnalysisResult, PredictResponse } from "../types/api";

export type AnalysisStatus = "idle" | "uploading" | "processing" | "done" | "error";

export interface UploadProgress {
  loaded: number;
  total: number;
}

interface AnalysisContextValue {
  status: AnalysisStatus;
  selectedFile: File | null;
  uploadProgress: UploadProgress;
  result: AnalysisResult | null;
  error: string | null;
  startAnalysis: (file: File) => Promise<void>;
  cancelAnalysis: () => void;
  reset: () => void;
}

const AnalysisContext = createContext<AnalysisContextValue | undefined>(undefined);

export function AnalysisProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<AnalysisStatus>("idle");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress>({ loaded: 0, total: 0 });
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  const startAnalysis = useCallback(async (file: File) => {
    setError(null);
    setResult(null);
    setSelectedFile(file);
    setUploadProgress({ loaded: 0, total: file.size });
    setStatus("uploading");

    const controller = new AbortController();
    abortControllerRef.current = controller;
    const startedAt = performance.now();

    try {
      const prediction: PredictResponse = await predictVideo(
        file,
        (loaded, total) => {
          setUploadProgress({ loaded, total });
          if (loaded >= total) setStatus("processing");
        },
        controller.signal,
      );
      const requestDurationMs = performance.now() - startedAt;
      setResult({
        prediction,
        sourceFileName: file.name,
        requestDurationMs,
        completedAt: new Date().toISOString(),
      });
      setStatus("done");
    } catch (err) {
      if (axiosWasCancelled(err)) return;
      const message = err instanceof ApiError ? err.message : "An unexpected error occurred while analyzing the video.";
      setError(message);
      setStatus("error");
    }
  }, []);

  const cancelAnalysis = useCallback(() => {
    abortControllerRef.current?.abort();
    setStatus("idle");
    setUploadProgress({ loaded: 0, total: 0 });
  }, []);

  const reset = useCallback(() => {
    abortControllerRef.current?.abort();
    setStatus("idle");
    setSelectedFile(null);
    setUploadProgress({ loaded: 0, total: 0 });
    setResult(null);
    setError(null);
  }, []);

  return (
    <AnalysisContext.Provider
      value={{ status, selectedFile, uploadProgress, result, error, startAnalysis, cancelAnalysis, reset }}
    >
      {children}
    </AnalysisContext.Provider>
  );
}

function axiosWasCancelled(err: unknown): boolean {
  return (
    typeof err === "object" &&
    err !== null &&
    (("name" in err && (err as { name?: string }).name === "CanceledError") ||
      ("code" in err && (err as { code?: string }).code === "ERR_CANCELED"))
  );
}

export function useAnalysis(): AnalysisContextValue {
  const ctx = useContext(AnalysisContext);
  if (!ctx) throw new Error("useAnalysis must be used within an AnalysisProvider");
  return ctx;
}
