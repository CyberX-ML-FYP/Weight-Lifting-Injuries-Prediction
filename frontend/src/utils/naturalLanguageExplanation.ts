import type { PredictResponse } from "../types/api";

// Template-based natural-language explanation, generated entirely on the
// client from the real values already returned by POST /predict. This does
// NOT reimplement, duplicate, or override any backend scoring/decision
// logic — it only phrases the numbers the backend already computed, using
// simple, clearly-labelled display bands (>=70 / 40-69 / <40) purely for
// human-readable wording. The backend's own adaptive thresholds
// (models/hip_knee_thresholds.json) remain the sole source of the actual
// Good/Poor classification and Rule A-D scores.

const RULE_LABELS: Record<"rule_a" | "rule_b" | "rule_c" | "rule_d", string> = {
  rule_a: "sequential extension (Rule A)",
  rule_b: "hip dominance (Rule B)",
  rule_c: "hip-knee synchronization (Rule C)",
  rule_d: "explosive movement / RFD approximation (Rule D)",
};

function band(score: number): "strong" | "moderate" | "weak" {
  if (score >= 70) return "strong";
  if (score >= 40) return "moderate";
  return "weak";
}

/**
 * Builds a single natural-language paragraph in the style requested by the
 * spec (e.g. "The athlete demonstrated good hip-knee coordination..."),
 * derived only from real returned fields.
 */
export function buildNaturalLanguageExplanation(p: PredictResponse): string {
  const cls = p.prediction;
  const ruleEntries: Array<["rule_a" | "rule_b" | "rule_c" | "rule_d", number]> = [
    ["rule_a", p.rule_a],
    ["rule_b", p.rule_b],
    ["rule_c", p.rule_c],
    ["rule_d", p.rule_d],
  ];
  const weakest = [...ruleEntries].sort((a, b) => a[1] - b[1])[0];
  const strongest = [...ruleEntries].sort((a, b) => b[1] - a[1])[0];

  const syncPhrase =
    band(p.rule_c) === "strong"
      ? "well-synchronized hip-knee coordination"
      : band(p.rule_c) === "moderate"
        ? "moderately synchronized hip-knee coordination with some timing delay"
        : "poorly synchronized hip-knee coordination with a notable timing delay";

  const romPhrase =
    p.hip_rom >= 60
      ? "a full hip range of motion"
      : "a hip range of motion that remained somewhat limited";

  const dominancePhrase =
    band(p.rule_b) === "strong"
      ? "balanced hip contribution relative to the knees"
      : band(p.rule_b) === "moderate"
        ? "a moderately hip-dominant movement pattern"
        : "an imbalanced hip/knee contribution pattern";

  return (
    `The model classified this lift as "${cls}" (${(p.prediction_confidence * 100).toFixed(1)}% confidence). ` +
    `The athlete demonstrated ${syncPhrase}. Hip range of motion ${p.hip_rom >= 60 ? "remained within" : "fell below"} ` +
    `typical acceptable limits (${romPhrase}, ${p.hip_rom.toFixed(1)}\u00B0). Rule B indicates ${dominancePhrase} ` +
    `(${p.rule_b.toFixed(0)}/100). Overall, ${RULE_LABELS[strongest[0]]} was this lift's strongest scored aspect ` +
    `(${strongest[1].toFixed(0)}/100), while ${RULE_LABELS[weakest[0]]} scored lowest ` +
    `(${weakest[1].toFixed(0)}/100) and is the most likely area to focus on next session.`
  );
}

/** Short bullet-style breakdown, one sentence per rule, for the explanation panel's detail list. */
export function buildRuleBreakdownSentences(p: PredictResponse): string[] {
  const lines: string[] = [];
  (Object.keys(RULE_LABELS) as Array<keyof typeof RULE_LABELS>).forEach((key) => {
    const score = p[key];
    const level = band(score);
    const label = RULE_LABELS[key];
    if (level === "strong") {
      lines.push(`${label[0].toUpperCase()}${label.slice(1)} looks strong (${score.toFixed(0)}/100).`);
    } else if (level === "moderate") {
      lines.push(`${label[0].toUpperCase()}${label.slice(1)} is moderate (${score.toFixed(0)}/100) — some room to improve.`);
    } else {
      lines.push(`${label[0].toUpperCase()}${label.slice(1)} scored low (${score.toFixed(0)}/100) and is a priority focus area.`);
    }
  });
  if (p.n_frames_used < 15) {
    lines.push(
      `Only ${p.n_frames_used} usable pose-quality frames were available — consider re-recording with clearer visibility of the hips/knees/ankles.`,
    );
  }
  lines.push(
    "This explanation is generated locally from the model's real returned scores and does not represent a certified coaching assessment.",
  );
  return lines;
}
