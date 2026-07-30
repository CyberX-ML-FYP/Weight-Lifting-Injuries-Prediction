import { useEffect, useRef, useState } from "react";
import { checkHealth } from "../services/apiClient";
import type { BackendStatus } from "../types/api";

const POLL_INTERVAL_MS = 15000;

/** Polls GET /health (the existing, unmodified endpoint) to drive the navbar's live backend-status indicator. */
export function useHealthCheck(): BackendStatus {
  const [status, setStatus] = useState<BackendStatus>("unknown");
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;

    async function poll() {
      try {
        const result = await checkHealth();
        if (!mountedRef.current) return;
        setStatus(result.status === "ok" ? "ok" : "starting");
      } catch {
        if (!mountedRef.current) return;
        setStatus("unreachable");
      }
    }

    poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      mountedRef.current = false;
      clearInterval(interval);
    };
  }, []);

  return status;
}
