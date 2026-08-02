// Client-side constants mirroring the backend's real, unmodified config
// (src/features/hip_knee_config.py: VIDEO_EXTENSIONS covers .mp4/.mov; this
// UI additionally lists .avi per the spec's "supported formats" list, but the
// backend will still reject it with its own 422 if unsupported — this is
// only a fast, friendly pre-check, never a replacement for server-side
// validation) and API limit (src/api/hip_knee_api.py: MAX_UPLOAD_BYTES = 500 MB).
export const ACCEPTED_EXTENSIONS = [".mp4", ".avi", ".mov"];
export const MAX_UPLOAD_BYTES = 500 * 1024 * 1024; // 500 MB

export interface FileValidationResult {
  valid: boolean;
  reason?: string;
}

export function validateVideoFile(file: File | null | undefined): FileValidationResult {
  if (!file) {
    return { valid: false, reason: "No file selected." };
  }
  const name = file.name || "";
  const ext = name.slice(name.lastIndexOf(".")).toLowerCase();
  if (!ACCEPTED_EXTENSIONS.includes(ext)) {
    return {
      valid: false,
      reason: `Unsupported file type "${ext || "(none)"}". Accepted types: ${ACCEPTED_EXTENSIONS.join(", ")}.`,
    };
  }
  if (file.size === 0) {
    return { valid: false, reason: "The selected file is empty (0 bytes)." };
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return {
      valid: false,
      reason: `File is too large (${formatBytes(file.size)}). Maximum allowed size is ${formatBytes(MAX_UPLOAD_BYTES)}.`,
    };
  }
  return { valid: true };
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / Math.pow(1024, exponent);
  return `${value.toFixed(exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}
