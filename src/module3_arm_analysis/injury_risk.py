"""
Module 3 — Injury Risk Interpretation

Translates measured biomechanical features (lockout completeness,
left/right symmetry) into a plain-language risk summary, using the same
thresholds analyzer.py already uses to flag frames (SYMMETRY_THRESHOLD,
LOCKOUT_THRESHOLD in config.py) -- not new, invented numbers.

IMPORTANT: this is a rule-based educational interpretation layer, not a
medical diagnosis. There is no injury-outcome data in this project (only
good/bad technique labels), so this does NOT predict a specific injury --
it flags technique patterns that weightlifting coaching / strength
literature commonly associates with injury risk, for explanatory purposes.

Author: Pasindu (214027H)
"""
import re

import pandas as pd

from src.module3_arm_analysis.config import LOCKOUT_THRESHOLD, SYMMETRY_THRESHOLD


def _first_present(features, *names):
    """Return (value, source_name) for the first non-NaN feature among names."""
    for name in names:
        value = features.get(name)
        if value is not None and not pd.isna(value):
            return value, name
    return None, None


def assess_injury_risk(features):
    """
    Inspect a lift's computed features and return a plain-language,
    rule-based injury-risk summary.

    Args:
        features: flat dict of computed feature name -> value, as produced
            by predict.py / feature_extractor.py (any subset of views is OK;
            checks that need missing data are simply skipped).

    Returns:
        dict with:
          risk_level: "Low" / "Moderate" / "High" / "Unknown" (if no
              checks could run at all, e.g. no side/angle view provided)
          checks: every check that was run, each a dict with
              title / status ("pass"/"fail") / detail / injury_note
          factors: the subset of checks with status "fail"
    """
    checks = []

    left, _ = _first_present(features, "side_lockout_left_elbow", "angle_lockout_left_elbow")
    right, _ = _first_present(features, "side_lockout_right_elbow", "angle_lockout_right_elbow")

    if left is not None and right is not None:
        # ── Check 1: full lockout ───────────────────────────────────────────
        min_elbow = min(left, right)
        lockout_failed = min_elbow < LOCKOUT_THRESHOLD
        checks.append({
            "title": "Full elbow lockout overhead",
            "status": "fail" if lockout_failed else "pass",
            "detail": (
                f"Left {left:.0f}°, right {right:.0f}° at the bar-overhead moment "
                f"(both need ≥ {LOCKOUT_THRESHOLD:.0f}° to count as fully locked out)."
            ),
            "injury_note": (
                "Repeatedly failing to fully extend the arms overhead under load is a "
                "well-documented technique fault in weightlifting coaching. It has been "
                "associated with shoulder impingement and added strain on the elbow/wrist "
                "from supporting the bar in a partially bent position."
            ) if lockout_failed else (
                "Both arms reached full extension overhead -- no added joint strain from "
                "supporting the bar in a bent position."
            ),
        })

        # ── Check 2: arms parallel (momentary symmetry, at lockout) ─────────
        diff = abs(left - right)
        parallel_failed = diff > SYMMETRY_THRESHOLD
        checks.append({
            "title": "Arms parallel overhead (left/right elbow match at lockout)",
            "status": "fail" if parallel_failed else "pass",
            "detail": f"{diff:.0f}° difference between left and right elbow (threshold {SYMMETRY_THRESHOLD:.0f}°).",
            "injury_note": (
                "One arm locking out noticeably more than the other means that arm is "
                "carrying more of the load. Bilateral asymmetry like this is a recognised "
                "risk factor for one-sided (unilateral) shoulder or elbow overuse injury "
                "if it happens repeatedly."
            ) if parallel_failed else (
                "Both arms extended to a matching angle -- the load appears evenly shared."
            ),
        })

    # ── Check 3: front-camera symmetry check AT the lockout moment ──────────
    # Note: this deliberately checks symmetry only at the lockout instant,
    # not averaged over the whole video. A whole-lift average includes
    # unrelated moments (walking up, adjusting the bar) and, when checked
    # against this project's own labeled data, does NOT distinguish good
    # from bad lifts (good/bad whole-lift averages are ~12° either way).
    # The lockout-moment measurement does (good ~4.5° vs bad ~14.7°), and
    # gives an independent, second-camera cross-check on the same lockout
    # moment Check 2 already looks at from the side/angle camera.
    front_lockout_sym = features.get("front_lockout_symmetry")
    if front_lockout_sym is not None and not pd.isna(front_lockout_sym):
        front_failed = front_lockout_sym > SYMMETRY_THRESHOLD
        checks.append({
            "title": "Front-camera symmetry check at lockout",
            "status": "fail" if front_failed else "pass",
            "detail": (
                f"{front_lockout_sym:.0f}° left/right difference at the lockout moment, "
                f"seen from the front camera (threshold {SYMMETRY_THRESHOLD:.0f}°)."
            ),
            "injury_note": (
                "A second camera angle confirms uneven arm extension at lockout, "
                "reinforcing the same one-sided loading concern as the parallel-arms "
                "check above."
            ) if front_failed else (
                "The front camera confirms the arms extended symmetrically at lockout."
            ),
        })

    if not checks:
        return {"risk_level": "Unknown", "checks": [], "factors": []}

    failed = [c for c in checks if c["status"] == "fail"]
    if len(failed) == 0:
        risk_level = "Low"
    elif len(failed) == 1:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return {"risk_level": risk_level, "checks": checks, "factors": failed}


# ── Friendly names for engineered feature columns (for UI display) ─────────

_VIEW_LABELS = {"side": "Side camera", "front": "Front camera", "angle": "Angle camera"}
_PHASE_LABELS = {"early": "first third", "middle": "middle third", "final": "final third"}
_STAT_LABELS = {"avg": "average", "max": "most-extended", "min": "most-bent"}


def friendly_feature_name(name):
    """Turn an engineered column name (e.g. 'angle_lockout_left_elbow') into
    a plain-language description, for showing feature values to a non-technical
    audience. Falls back to a prettified version of the raw name if the
    pattern isn't recognized."""
    for view, view_label in _VIEW_LABELS.items():
        prefix = f"{view}_"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]

        m = re.match(r"^(early|middle|final)_(avg|max|min)_(left|right)_elbow$", rest)
        if m:
            phase, stat, side = m.groups()
            return f"{view_label}: {_STAT_LABELS[stat]} {side} elbow angle, {_PHASE_LABELS[phase]} of the lift"

        m = re.match(r"^std_(left|right)_elbow$", rest)
        if m:
            return f"{view_label}: {m.group(1)} elbow angle variability across the whole lift"

        m = re.match(r"^elbow_range_(left|right)$", rest)
        if m:
            return f"{view_label}: {m.group(1)} elbow's full range of motion (most-extended minus most-bent)"

        if rest == "lockout_ratio":
            return f"{view_label}: fraction of the lift spent with incomplete lockout"
        if rest == "lockout_frame_ratio":
            return f"{view_label}: how far through the lift the lockout moment occurred"
        if rest in ("lockout_left_elbow", "lockout_right_elbow"):
            side = rest.split("_")[1]
            return f"{view_label}: {side} elbow angle at the lockout moment"

    if name.startswith("front_"):
        rest = name[len("front_"):]
        m = re.match(r"^(early|middle|final)_avg_symmetry_diff$", rest)
        if m:
            return f"Front camera: average left/right symmetry difference, {_PHASE_LABELS[m.group(1)]} of the lift"
        if rest == "overall_avg_symmetry":
            return "Front camera: average left/right symmetry difference, whole lift"
        if rest == "overall_max_symmetry":
            return "Front camera: worst-moment left/right symmetry difference"
        if rest == "asymmetry_ratio":
            return "Front camera: fraction of the lift flagged as asymmetric"
        if rest == "lockout_symmetry":
            return "Front camera: left/right symmetry difference at the lockout moment"

    return name.replace("_", " ").capitalize()
