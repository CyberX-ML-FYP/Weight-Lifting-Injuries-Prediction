import type { AnalysisResult } from "../types/api";

/** Trigger a browser download of arbitrary text content (no backend round-trip). */
export function downloadTextFile(filename: string, content: string, mimeType = "application/octet-stream"): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/** Download the *actual* live prediction response returned by the backend for this request. */
export function downloadPredictionJson(result: AnalysisResult): void {
  const payload = JSON.stringify(result.prediction, null, 2);
  const base = (result.sourceFileName || "prediction").replace(/\.[^/.]+$/, "");
  downloadTextFile(`${base}_prediction.json`, payload, "application/json");
}

/**
 * Build a CSV of the real Rule A-D scores + confidences from this exact
 * prediction (live data, not a mocked/sample file) — a lightweight
 * "Explainability CSV" derived only from already-returned fields.
 */
export function downloadExplainabilityCsv(result: AnalysisResult): void {
  const p = result.prediction;
  const rows = [
    ["metric", "value"],
    ["prediction", p.prediction],
    ["prediction_confidence", p.prediction_confidence.toString()],
    ["low_confidence_flag", p.low_confidence_flag.toString()],
    ["rule_a", p.rule_a.toString()],
    ["rule_b", p.rule_b.toString()],
    ["rule_c", p.rule_c.toString()],
    ["rule_d", p.rule_d.toString()],
    ["confidence_weighted_score", p.confidence_weighted_score.toString()],
    ["rule_a_confidence", p.rule_confidences.rule_a_confidence.toString()],
    ["rule_b_confidence", p.rule_confidences.rule_b_confidence.toString()],
    ["rule_c_confidence", p.rule_confidences.rule_c_confidence.toString()],
    ["rule_d_confidence", p.rule_confidences.rule_d_confidence.toString()],
    ["overall_confidence", p.rule_confidences.overall_confidence.toString()],
    ["hip_rom_deg", p.hip_rom.toString()],
    ["knee_rom_deg", p.knee_rom.toString()],
    ["correlation", p.correlation.toString()],
    ["rfd", p.rfd.toString()],
  ];
  const csv = rows.map((r) => r.map((cell) => `"${cell}"`).join(",")).join("\n");
  const base = (result.sourceFileName || "prediction").replace(/\.[^/.]+$/, "");
  downloadTextFile(`${base}_explainability.csv`, csv, "text/csv");
}
