"""
6-month aerobic base training plan.

Profile: 28yo, max HR 216, no aerobic base, HR spikes to 180 instantly.
MAF target: 180 - 28 - 10 = 142 bpm
Zone 2: 130-151 bpm
Working target: 137-147 bpm

3 phases, 26 weeks, 4 sessions/week.
"""

import json
import math
from datetime import date, timedelta

import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────
PLAN_START = date(2026, 3, 30)  # Monday of week 1
AGE = 28
MAX_HR = 216
MAF_TARGET = 180 - AGE - 10  # 142
ZONE2_LO = 130
ZONE2_HI = 151
TARGET_LO = 137
TARGET_HI = 147

SESSION_HR_TARGETS = {
    "cycling_z2":  (ZONE2_LO, ZONE2_HI),
    "run_walk":    (TARGET_LO, TARGET_HI),
    "easy_run":    (TARGET_LO, TARGET_HI),
    "tempo_run":   (155, 170),
    "rest":        (0, 0),
}

SESSION_LABELS = {
    "cycling_z2": "Cycling Zone 2",
    "run_walk":   "Run/Walk Intervals",
    "easy_run":   "Easy Continuous Run",
    "tempo_run":  "Tempo Run",
    "rest":       "Rest Day",
}


def _foundation_sessions(week):
    """Weeks 1-8: 2x cycling + 2x run/walk."""
    # Progressive overload within foundation
    cycling_min = 30 + min(week - 1, 7) * 1.5  # 30 → 40.5
    rw_total_min = 20 + min(week - 1, 7) * 1.5  # 20 → 30.5
    run_seg = 1 + min(week - 1, 7) * 0.4  # 1 → 3.8 min
    walk_seg = 3 - min(week - 1, 7) * 0.25  # 3 → 1.25 min

    return [
        {"day": "Monday",    "type": "cycling_z2", "duration_min": round(cycling_min),
         "notes": f"Steady cycling, HR {ZONE2_LO}-{ZONE2_HI}"},
        {"day": "Tuesday",   "type": "rest",       "duration_min": 0, "notes": "Recovery"},
        {"day": "Wednesday", "type": "run_walk",    "duration_min": round(rw_total_min),
         "notes": f"Run {run_seg:.0f}min / Walk {walk_seg:.0f}min, HR <{TARGET_HI}"},
        {"day": "Thursday",  "type": "rest",        "duration_min": 0, "notes": "Recovery"},
        {"day": "Friday",    "type": "cycling_z2",  "duration_min": round(cycling_min),
         "notes": f"Steady cycling, HR {ZONE2_LO}-{ZONE2_HI}"},
        {"day": "Saturday",  "type": "run_walk",    "duration_min": round(rw_total_min),
         "notes": f"Run {run_seg:.0f}min / Walk {walk_seg:.0f}min, HR <{TARGET_HI}"},
        {"day": "Sunday",    "type": "rest",        "duration_min": 0, "notes": "Recovery"},
    ]


def _building_sessions(week):
    """Weeks 9-17: 1x cycling + 3x run/walk (longer runs)."""
    w = week - 9  # 0-8
    cycling_min = 35 + w * 1.5  # 35 → 47
    rw_total_min = 30 + w * 2.5  # 30 → 50
    run_seg = 5 + w * 1.0  # 5 → 13 min segments
    walk_seg = max(1, 2 - w * 0.12)  # 2 → ~1 min

    return [
        {"day": "Monday",    "type": "run_walk",    "duration_min": round(rw_total_min),
         "notes": f"Run {run_seg:.0f}min / Walk {walk_seg:.0f}min, HR <{TARGET_HI}"},
        {"day": "Tuesday",   "type": "rest",        "duration_min": 0, "notes": "Recovery"},
        {"day": "Wednesday", "type": "cycling_z2",   "duration_min": round(cycling_min),
         "notes": f"Steady cycling, HR {ZONE2_LO}-{ZONE2_HI}"},
        {"day": "Thursday",  "type": "rest",         "duration_min": 0, "notes": "Recovery"},
        {"day": "Friday",    "type": "run_walk",     "duration_min": round(rw_total_min),
         "notes": f"Run {run_seg:.0f}min / Walk {walk_seg:.0f}min, HR <{TARGET_HI}"},
        {"day": "Saturday",  "type": "run_walk",     "duration_min": round(rw_total_min + 5),
         "notes": f"Long run/walk, HR <{TARGET_HI}"},
        {"day": "Sunday",    "type": "rest",         "duration_min": 0, "notes": "Recovery"},
    ]


def _development_sessions(week):
    """Weeks 18-26: 3x easy continuous runs + 1x tempo."""
    w = week - 18  # 0-8
    easy_min = 30 + w * 2.5  # 30 → 50
    tempo_min = 20 + w * 1.5  # 20 → 32

    return [
        {"day": "Monday",    "type": "easy_run",   "duration_min": round(easy_min),
         "notes": f"Easy run, HR {TARGET_LO}-{TARGET_HI}"},
        {"day": "Tuesday",   "type": "rest",       "duration_min": 0, "notes": "Recovery"},
        {"day": "Wednesday", "type": "tempo_run",   "duration_min": round(tempo_min),
         "notes": "Tempo run, HR 155-170"},
        {"day": "Thursday",  "type": "rest",        "duration_min": 0, "notes": "Recovery"},
        {"day": "Friday",    "type": "easy_run",    "duration_min": round(easy_min),
         "notes": f"Easy run, HR {TARGET_LO}-{TARGET_HI}"},
        {"day": "Saturday",  "type": "easy_run",    "duration_min": round(easy_min + 10),
         "notes": f"Long easy run, HR {TARGET_LO}-{TARGET_HI}"},
        {"day": "Sunday",    "type": "rest",        "duration_min": 0, "notes": "Recovery"},
    ]


def generate_plan():
    """Generate the full 26-week plan as a list of week dicts."""
    plan = []
    for week_num in range(1, 27):
        if week_num <= 8:
            phase = "Foundation"
            sessions = _foundation_sessions(week_num)
        elif week_num <= 17:
            phase = "Building"
            sessions = _building_sessions(week_num)
        else:
            phase = "Development"
            sessions = _development_sessions(week_num)

        week_start = PLAN_START + timedelta(weeks=week_num - 1)
        for i, s in enumerate(sessions):
            s["date"] = (week_start + timedelta(days=i)).isoformat()
            s["hr_target"] = SESSION_HR_TARGETS[s["type"]]

        plan.append({
            "week": week_num,
            "phase": phase,
            "start_date": week_start.isoformat(),
            "end_date": (week_start + timedelta(days=6)).isoformat(),
            "sessions": sessions,
        })
    return plan


def get_current_week(plan):
    """Return the current week dict based on today's date, or None."""
    today = date.today()
    for w in plan:
        ws = date.fromisoformat(w["start_date"])
        we = date.fromisoformat(w["end_date"])
        if ws <= today <= we:
            return w
    return None


def compute_adherence(plan, activities_df):
    """
    Compare planned sessions against actual Strava activities.

    activities_df should have columns: date, type (Run/Ride),
    average_heartrate, moving_time_min.

    Returns a list of week-level adherence records.
    """
    if activities_df is None or activities_df.empty:
        return []

    # Normalize dates
    if not pd.api.types.is_datetime64_any_dtype(activities_df["date"]):
        activities_df = activities_df.copy()
        activities_df["date"] = pd.to_datetime(activities_df["date"])

    adherence = []
    for week in plan:
        planned = 0
        completed = 0
        hr_compliant = 0

        for session in week["sessions"]:
            if session["type"] == "rest":
                continue
            planned += 1

            sess_date = pd.Timestamp(session["date"])
            hr_lo, hr_hi = session["hr_target"]

            # Match by date (±1 day tolerance) and activity type
            if session["type"] == "cycling_z2":
                match_type = "Ride"
            else:
                match_type = "Run"

            window = activities_df[
                (activities_df["date"] >= sess_date - timedelta(days=1)) &
                (activities_df["date"] <= sess_date + timedelta(days=1))
            ]
            if "activity_type" in window.columns:
                matches = window[window["activity_type"] == match_type]
            else:
                matches = window

            if not matches.empty:
                completed += 1
                avg_hr = matches.iloc[0].get("average_heartrate")
                if avg_hr and hr_lo <= avg_hr <= hr_hi:
                    hr_compliant += 1

        adherence.append({
            "week": week["week"],
            "phase": week["phase"],
            "start_date": week["start_date"],
            "planned": planned,
            "completed": completed,
            "hr_compliant": hr_compliant,
            "completion_pct": round(100 * completed / planned, 1) if planned > 0 else 0,
            "hr_compliance_pct": round(100 * hr_compliant / planned, 1) if planned > 0 else 0,
        })

    return adherence


def plan_to_json(plan):
    """Serialize plan to JSON-safe format."""
    out = []
    for week in plan:
        w = {**week}
        w["sessions"] = []
        for s in week["sessions"]:
            sess = {**s}
            sess["hr_target"] = list(sess["hr_target"])
            w["sessions"].append(sess)
        out.append(w)
    return out


if __name__ == "__main__":
    plan = generate_plan()
    print(json.dumps(plan_to_json(plan), indent=2))
    current = get_current_week(plan)
    if current:
        print(f"\nCurrent week: {current['week']} ({current['phase']})")
    else:
        print("\nPlan hasn't started yet or has ended.")
