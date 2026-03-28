"""
Build a self-contained HTML dashboard from Strava run data.
Reads streams_data.json and runs_panel.csv, outputs dashboard.html
"""

import json
import math
import os

import pandas as pd

from training_plan import (
    generate_plan, get_current_week, compute_adherence, plan_to_json,
    SESSION_LABELS, PLAN_START, TARGET_LO, TARGET_HI, MAF_TARGET,
)

BASE_DIR = os.path.dirname(__file__)
STREAMS_FILE = os.path.join(BASE_DIR, "streams_data.json")
PANEL_FILE = os.path.join(BASE_DIR, "runs_panel.csv")
DASHBOARD_FILE = os.path.join(BASE_DIR, "dashboard.html")


def downsample(arr, max_points=300):
    """Keep at most max_points evenly spaced."""
    if len(arr) <= max_points:
        return arr
    step = len(arr) / max_points
    return [arr[int(i * step)] for i in range(max_points)]


def compute_splits(streams, split_dist=1000):
    """Compute per-km splits from distance/time/hr streams."""
    dist = streams.get("distance", [])
    time = streams.get("time", [])
    hr = streams.get("heartrate", [])
    if not dist or not time:
        return []
    splits = []
    km = 1
    prev_time = 0
    prev_dist = 0
    hr_accum = []
    for i in range(len(dist)):
        if hr and i < len(hr):
            hr_accum.append(hr[i])
        if dist[i] >= km * split_dist:
            dt = time[i] - prev_time
            dd = dist[i] - prev_dist
            pace = (dt / 60) / (dd / 1000) if dd > 0 else 0
            avg_hr = sum(hr_accum) / len(hr_accum) if hr_accum else None
            splits.append({
                "km": km,
                "pace_min_km": round(pace, 2),
                "time_s": dt,
                "avg_hr": round(avg_hr) if avg_hr else None,
            })
            km += 1
            prev_time = time[i]
            prev_dist = dist[i]
            hr_accum = []
    return splits


def build():
    """Build the complete dashboard HTML."""
    # ── Load data ──────────────────────────────────────────────────────────
    with open(STREAMS_FILE) as f:
        streams_raw = json.load(f)

    panel = pd.read_csv(PANEL_FILE, parse_dates=["date"])

    # ── Training plan ──────────────────────────────────────────────────────
    plan = generate_plan()
    current_week = get_current_week(plan)
    adherence = compute_adherence(plan, panel)
    plan_json = plan_to_json(plan)

    # ── Prepare compact data for embedding ─────────────────────────────────
    # Filter to only runs for stream charts (rides don't have same stream types)
    runs_data = []
    for aid, act in streams_raw.items():
        s = act["streams"]
        has_hr = "heartrate" in s and len(s["heartrate"]) > 10
        has_vel = "velocity_smooth" in s and len(s["velocity_smooth"]) > 10

        run = {
            "id": act["activity_id"],
            "date": act["date"],
            "name": act["name"],
            "distance_km": act["distance_km"],
            "time": downsample(s.get("time", [])),
            "altitude": downsample(s.get("altitude", [])),
            "heartrate": downsample(s.get("heartrate", [])) if has_hr else [],
            "velocity": downsample(s.get("velocity_smooth", [])) if has_vel else [],
            "grade": downsample(s.get("grade_smooth", [])),
            "splits": compute_splits(s),
        }
        runs_data.append(run)

    runs_data.sort(key=lambda r: r["date"])

    # Panel summary data
    panel_records = panel.to_dict("records")
    for r in panel_records:
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                r[k] = None
            elif hasattr(v, "isoformat"):
                r[k] = v.isoformat()

    # ── HR Zone computation ────────────────────────────────────────────────
    max_hr_observed = panel["max_heartrate"].max()

    def hr_zone_bounds(max_hr):
        return [
            (0, 0.60 * max_hr, "Zone 1 — Recovery"),
            (0.60 * max_hr, 0.70 * max_hr, "Zone 2 — Easy"),
            (0.70 * max_hr, 0.80 * max_hr, "Zone 3 — Aerobic"),
            (0.80 * max_hr, 0.90 * max_hr, "Zone 4 — Threshold"),
            (0.90 * max_hr, max_hr * 1.1, "Zone 5 — Max"),
        ]

    zone_bounds = hr_zone_bounds(max_hr_observed)

    # Compute zone time per run from full (non-downsampled) streams
    zone_data_per_run = []
    for aid, act in streams_raw.items():
        hr = act["streams"].get("heartrate", [])
        time_s = act["streams"].get("time", [])
        if len(hr) < 10 or len(time_s) < 10:
            continue
        zones = [0, 0, 0, 0, 0]
        for i in range(1, min(len(hr), len(time_s))):
            dt = time_s[i] - time_s[i - 1]
            for z, (lo, hi, _) in enumerate(zone_bounds):
                if lo <= hr[i] < hi:
                    zones[z] += dt
                    break
        total = sum(zones)
        if total > 0:
            zone_data_per_run.append({
                "date": act["date"],
                "name": act["name"],
                "zones_pct": [round(100 * z / total, 1) for z in zones],
                "zones_sec": zones,
                "total_sec": total,
            })

    zone_data_per_run.sort(key=lambda x: x["date"])

    # Monthly volume (runs only for pace stats)
    runs_only = panel[panel["activity_type"] == "Run"] if "activity_type" in panel.columns else panel
    runs_only = runs_only.copy()
    runs_only["month_label"] = runs_only["date"].dt.to_period("M").astype(str)
    monthly = runs_only.groupby("month_label").agg(
        total_km=("distance_km", "sum"),
        runs=("distance_km", "count"),
        avg_pace=("pace_min_per_km", "mean"),
        avg_hr=("average_heartrate", "mean"),
        total_time_min=("moving_time_min", "sum"),
    ).reset_index()
    monthly_records = monthly.to_dict("records")
    for r in monthly_records:
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                r[k] = None

    # ── Generate HTML ──────────────────────────────────────────────────────

    total_km = round(runs_only["distance_km"].sum(), 1)
    total_runs = len(runs_only)
    total_hours = round(runs_only["moving_time_min"].sum() / 60, 1)
    avg_pace = runs_only.loc[runs_only["pace_min_per_km"] < 20, "pace_min_per_km"].mean()
    avg_hr = runs_only["average_heartrate"].mean()
    longest_run = runs_only["distance_km"].max()

    # Current week info for the template
    cw_num = current_week["week"] if current_week else 0
    cw_phase = current_week["phase"] if current_week else "Not started"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Run Panel — Strava Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg-deep: #0a0a0c;
  --bg-card: #111115;
  --bg-card-hover: #18181e;
  --border: #222230;
  --text-primary: #e8e6e3;
  --text-secondary: #7a7880;
  --text-muted: #4a4850;
  --accent-coral: #ff6b4a;
  --accent-amber: #f5a623;
  --accent-cyan: #4ad4e0;
  --accent-violet: #9b7aff;
  --accent-rose: #ff4a7a;
  --accent-green: #22c55e;
  --zone1: #3b82f6;
  --zone2: #22c55e;
  --zone3: #f5a623;
  --zone4: #ff6b4a;
  --zone5: #ff4a7a;
  --radius: 10px;
  --font-body: 'Instrument Sans', sans-serif;
  --font-mono: 'Space Mono', monospace;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
  background: var(--bg-deep);
  color: var(--text-primary);
  font-family: var(--font-body);
  line-height: 1.5;
  min-height: 100vh;
}}

/* Grain overlay */
body::before {{
  content: '';
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 9999;
}}

.dashboard {{
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px 24px;
}}

/* ── Header ─────────────────────────── */
.header {{
  display: flex;
  align-items: baseline;
  gap: 16px;
  margin-bottom: 48px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 24px;
}}
.header h1 {{
  font-family: var(--font-mono);
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, var(--accent-coral), var(--accent-amber));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.header .subtitle {{
  font-size: 13px;
  color: var(--text-muted);
  font-family: var(--font-mono);
}}

/* ── Stat Cards ─────────────────────── */
.stats-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-bottom: 40px;
}}
.stat-card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}}
.stat-card:hover {{
  border-color: var(--accent-coral);
}}
.stat-card::after {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent-coral), transparent);
  opacity: 0.6;
}}
.stat-label {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: var(--text-muted);
  font-family: var(--font-mono);
  margin-bottom: 8px;
}}
.stat-value {{
  font-size: 32px;
  font-weight: 700;
  font-family: var(--font-mono);
  color: var(--text-primary);
}}
.stat-unit {{
  font-size: 14px;
  color: var(--text-secondary);
  font-weight: 400;
}}

/* ── Section Layout ─────────────────── */
.section {{
  margin-bottom: 40px;
}}
.section-title {{
  font-family: var(--font-mono);
  font-size: 14px;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--text-muted);
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 10px;
}}
.section-title::after {{
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}}

.charts-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}}
@media (max-width: 900px) {{
  .charts-row {{ grid-template-columns: 1fr; }}
}}

.chart-card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  position: relative;
}}
.chart-card.full {{
  grid-column: 1 / -1;
}}
.chart-card h3 {{
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 16px;
  color: var(--text-primary);
}}
.chart-card canvas {{
  width: 100% !important;
}}

/* ── Run Selector ───────────────────── */
.run-selector {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}}
.run-btn {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 12px;
  color: var(--text-secondary);
  font-family: var(--font-mono);
  font-size: 11px;
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
}}
.run-btn:hover {{
  border-color: var(--accent-coral);
  color: var(--text-primary);
}}
.run-btn.active {{
  background: var(--accent-coral);
  border-color: var(--accent-coral);
  color: #000;
}}

select.run-dropdown {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 14px;
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 13px;
  width: 100%;
  max-width: 400px;
  margin-bottom: 16px;
  cursor: pointer;
  outline: none;
}}
select.run-dropdown:focus {{
  border-color: var(--accent-coral);
}}

/* ── Splits Table ───────────────────── */
.splits-table {{
  width: 100%;
  border-collapse: collapse;
  font-family: var(--font-mono);
  font-size: 13px;
}}
.splits-table th {{
  text-align: left;
  padding: 8px 12px;
  color: var(--text-muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1px;
  border-bottom: 1px solid var(--border);
}}
.splits-table td {{
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  color: var(--text-secondary);
}}
.splits-table tr:hover td {{
  color: var(--text-primary);
  background: var(--bg-card-hover);
}}
.pace-bar {{
  display: inline-block;
  height: 4px;
  background: var(--accent-coral);
  border-radius: 2px;
  margin-left: 8px;
  vertical-align: middle;
  opacity: 0.7;
}}

/* ── Training Plan ─────────────────── */
.tp-hero {{
  display: grid;
  grid-template-columns: 260px 1fr;
  gap: 20px;
  margin-bottom: 28px;
}}
@media (max-width: 960px) {{
  .tp-hero {{ grid-template-columns: 1fr; }}
}}
.tp-hero-left {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px 24px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  position: relative;
  overflow: hidden;
}}
.tp-hero-left::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--accent-coral), var(--accent-amber), var(--accent-cyan));
}}
.tp-ring-wrap {{
  position: relative;
  width: 120px;
  height: 120px;
}}
.tp-ring-wrap svg {{ transform: rotate(-90deg); }}
.tp-ring-label {{
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}}
.tp-ring-week {{
  font-family: var(--font-mono);
  font-size: 32px;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1;
}}
.tp-ring-sub {{
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-top: 4px;
}}
.tp-phase-name {{
  font-family: var(--font-mono);
  font-size: 16px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 3px;
}}
.tp-phase-foundation {{ color: #4ad4e0; }}
.tp-phase-building {{ color: #f5a623; }}
.tp-phase-development {{ color: #9b7aff; }}
.tp-phase-pending {{ color: var(--text-muted); }}
.tp-hero-meta {{
  display: flex;
  gap: 20px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-secondary);
}}
.tp-hero-meta span {{
  display: flex;
  align-items: center;
  gap: 6px;
}}
.tp-meta-dot {{
  width: 6px;
  height: 6px;
  border-radius: 50%;
  display: inline-block;
}}

/* Current week strip */
.tp-week-strip {{
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 6px;
  align-content: start;
}}
@media (max-width: 960px) {{
  .tp-week-strip {{ grid-template-columns: repeat(4, 1fr); }}
}}
@media (max-width: 500px) {{
  .tp-week-strip {{ grid-template-columns: repeat(2, 1fr); }}
}}
.tp-sc {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 10px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s, transform 0.2s;
}}
.tp-sc:hover {{ transform: translateY(-2px); }}
.tp-sc.is-today {{
  border-color: var(--accent-coral);
}}
.tp-sc.is-today::after {{
  content: 'TODAY';
  position: absolute;
  top: 4px; right: 6px;
  font-family: var(--font-mono);
  font-size: 8px;
  letter-spacing: 1px;
  color: var(--accent-coral);
  font-weight: 700;
}}
.tp-sc.is-rest {{ opacity: 0.4; }}
.tp-sc .tp-sc-bar {{
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
}}
.tp-sc-day {{
  font-family: var(--font-mono);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--text-muted);
  margin-bottom: 8px;
}}
.tp-sc-type {{
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
  line-height: 1.3;
}}
.tp-sc-dur {{
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-secondary);
}}
.tp-sc-hr {{
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-muted);
  margin-top: 4px;
}}

/* Timeline */
.tp-tl-wrap {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  margin-bottom: 20px;
}}
.tp-tl-row {{
  display: grid;
  grid-template-columns: repeat(26, 1fr);
  gap: 4px;
  margin-bottom: 10px;
}}
.tp-tl-cell {{
  height: 32px;
  border-radius: 4px;
  position: relative;
  cursor: pointer;
  transition: transform 0.15s;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-mono);
  font-size: 9px;
  color: transparent;
}}
.tp-tl-cell:hover {{
  transform: scaleY(1.4);
  color: var(--text-primary);
  z-index: 2;
}}
.tp-tl-cell.past {{
  border: 1px solid var(--border);
}}
.tp-tl-cell.future {{
  background: var(--bg-card-hover);
  border: 1px dashed #2a2a35;
}}
.tp-tl-cell.current {{
  border: 2px solid var(--accent-coral);
  animation: tp-pulse 2s ease-in-out infinite;
}}
@keyframes tp-pulse {{
  0%, 100% {{ box-shadow: 0 0 0 0 rgba(255,107,74,0.3); }}
  50% {{ box-shadow: 0 0 12px 2px rgba(255,107,74,0.2); }}
}}
.tp-tl-legend {{
  display: flex;
  justify-content: space-between;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-muted);
}}

/* Full Plan Accordion */
.tp-full-plan {{ margin-top: 8px; }}
.tp-pg {{
  margin-bottom: 12px;
}}
.tp-pg-hd {{
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px 8px 0 0;
  cursor: pointer;
  user-select: none;
  transition: background 0.15s;
}}
.tp-pg-hd:hover {{ background: var(--bg-card-hover); }}
.tp-pg-hd.collapsed {{ border-radius: 8px; }}
.tp-pg-bar {{
  width: 4px;
  height: 20px;
  border-radius: 2px;
}}
.tp-pg-title {{
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 2px;
  flex: 1;
}}
.tp-pg-range {{
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-muted);
}}
.tp-pg-chev {{
  color: var(--text-muted);
  transition: transform 0.25s;
  font-size: 12px;
}}
.tp-pg-hd.open .tp-pg-chev {{ transform: rotate(180deg); }}
.tp-pg-body {{
  border: 1px solid var(--border);
  border-top: none;
  border-radius: 0 0 8px 8px;
  overflow: hidden;
  max-height: 0;
  transition: max-height 0.4s ease;
}}
.tp-pg-body.open {{ max-height: 6000px; }}

.tp-wa {{
  border-bottom: 1px solid rgba(255,255,255,0.03);
}}
.tp-wa:last-child {{ border-bottom: none; }}
.tp-wa-hd {{
  display: grid;
  grid-template-columns: 48px 130px 1fr 70px 40px;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  cursor: pointer;
  transition: background 0.15s;
  font-family: var(--font-mono);
  font-size: 12px;
}}
@media (max-width: 600px) {{
  .tp-wa-hd {{ grid-template-columns: 48px 1fr 60px 40px; }}
  .tp-wa-hd .tp-wa-dates {{ display: none; }}
}}
.tp-wa-hd:hover {{ background: var(--bg-card-hover); }}
.tp-wa-hd.cw {{
  background: rgba(255,107,74,0.04);
  border-left: 3px solid var(--accent-coral);
}}
.tp-wa-num {{ font-weight: 700; color: var(--text-secondary); }}
.tp-wa-dates {{ color: var(--text-muted); font-size: 11px; }}
.tp-wa-prog {{
  height: 4px;
  background: #1a1a22;
  border-radius: 2px;
  overflow: hidden;
}}
.tp-wa-prog-fill {{
  height: 100%;
  border-radius: 2px;
  transition: width 0.3s;
}}
.tp-wa-pct {{
  text-align: right;
  color: var(--text-muted);
  font-size: 11px;
}}
.tp-wa-chev {{
  text-align: center;
  color: var(--text-muted);
  transition: transform 0.2s;
  font-size: 10px;
}}
.tp-wa-hd.open .tp-wa-chev {{ transform: rotate(180deg); }}
.tp-wa-det {{
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.3s ease;
  background: rgba(0,0,0,0.12);
}}
.tp-wa-det.open {{ max-height: 500px; }}
.tp-sg {{
  display: grid;
  gap: 0;
  padding: 6px 16px 10px;
}}
.tp-sr {{
  display: grid;
  grid-template-columns: 56px 1fr 64px 110px;
  gap: 12px;
  padding: 7px 0;
  border-bottom: 1px solid rgba(255,255,255,0.025);
  font-family: var(--font-mono);
  font-size: 11px;
  align-items: center;
}}
.tp-sr:last-child {{ border-bottom: none; }}
.tp-sr-day {{
  color: var(--text-muted);
  text-transform: uppercase;
  font-size: 10px;
  letter-spacing: 0.5px;
}}
.tp-sr-type {{
  display: flex;
  align-items: center;
  gap: 8px;
}}
.tp-sr-dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}}
.tp-sr-name {{
  color: var(--text-primary);
  font-weight: 500;
}}
.tp-sr-dur {{ color: var(--text-secondary); }}
.tp-sr-hr {{ color: var(--text-muted); font-size: 10px; }}

/* ── Animations ─────────────────────── */
@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(12px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
.stat-card, .chart-card {{
  animation: fadeUp 0.5s ease both;
}}
.stat-card:nth-child(1) {{ animation-delay: 0.05s; }}
.stat-card:nth-child(2) {{ animation-delay: 0.10s; }}
.stat-card:nth-child(3) {{ animation-delay: 0.15s; }}
.stat-card:nth-child(4) {{ animation-delay: 0.20s; }}
.stat-card:nth-child(5) {{ animation-delay: 0.25s; }}
.stat-card:nth-child(6) {{ animation-delay: 0.30s; }}

/* ── Zone Legend ─────────────────────── */
.zone-legend {{
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 12px;
  font-size: 12px;
  font-family: var(--font-mono);
}}
.zone-legend span {{
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--text-secondary);
}}
.zone-dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}}
</style>
</head>
<body>
<div class="dashboard">

  <div class="header">
    <h1>RUN//PANEL</h1>
    <span class="subtitle">{panel['date'].min().strftime('%b %Y')} — {panel['date'].max().strftime('%b %Y')} &middot; {total_runs} runs</span>
  </div>

  <!-- ── Overview Stats ───────────────── -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Total Distance</div>
      <div class="stat-value">{total_km}<span class="stat-unit"> km</span></div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total Runs</div>
      <div class="stat-value">{total_runs}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Hours Running</div>
      <div class="stat-value">{total_hours}<span class="stat-unit"> h</span></div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Avg Pace</div>
      <div class="stat-value">{avg_pace:.1f}<span class="stat-unit"> min/km</span></div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Avg Heart Rate</div>
      <div class="stat-value">{avg_hr:.0f}<span class="stat-unit"> bpm</span></div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Longest Run</div>
      <div class="stat-value">{longest_run}<span class="stat-unit"> km</span></div>
    </div>
  </div>

  <!-- ── Training Plan ─────────────────── -->
  <div class="section">
    <div class="section-title">Training Plan — 26 Week Aerobic Base</div>

    <!-- Hero: ring + current week strip -->
    <div class="tp-hero">
      <div class="tp-hero-left">
        <div class="tp-ring-wrap" id="tpRing"></div>
        <div class="tp-phase-name tp-phase-pending" id="tpPhaseName">{cw_phase}</div>
        <div class="tp-hero-meta">
          <span><span class="tp-meta-dot" style="background:var(--accent-coral)"></span>MAF {MAF_TARGET}</span>
          <span><span class="tp-meta-dot" style="background:var(--accent-green)"></span>Target {TARGET_LO}-{TARGET_HI}</span>
        </div>
      </div>
      <div id="tpWeekStrip" class="tp-week-strip"></div>
    </div>

    <!-- 26-week timeline -->
    <div class="tp-tl-wrap">
      <h3 style="font-size:14px;font-weight:600;margin-bottom:16px;color:var(--text-primary)">26-Week Timeline</h3>
      <div id="tpTimelineRow" class="tp-tl-row"></div>
      <div class="tp-tl-legend">
        <span style="color:#4ad4e0">Foundation (1-8)</span>
        <span style="color:#f5a623">Building (9-17)</span>
        <span style="color:#9b7aff">Development (18-26)</span>
      </div>
    </div>

    <!-- Adherence charts -->
    <div class="charts-row" style="margin-bottom:20px">
      <div class="chart-card">
        <h3>Weekly Adherence</h3>
        <canvas id="adherenceChart" height="260"></canvas>
      </div>
      <div class="chart-card">
        <h3>HR vs Target (Plan Sessions)</h3>
        <canvas id="hrTargetChart" height="260"></canvas>
      </div>
    </div>

    <!-- Full plan accordion -->
    <div class="chart-card full">
      <h3>Full Plan — Week by Week</h3>
      <div id="tpFullPlan" class="tp-full-plan"></div>
    </div>
  </div>

  <!-- ── Monthly Volume ───────────────── -->
  <div class="section">
    <div class="section-title">Monthly Volume</div>
    <div class="charts-row">
      <div class="chart-card">
        <h3>Distance & Runs per Month</h3>
        <canvas id="monthlyVolume" height="240"></canvas>
      </div>
      <div class="chart-card">
        <h3>Avg HR & Pace Trend</h3>
        <canvas id="monthlyTrend" height="240"></canvas>
      </div>
    </div>
  </div>

  <!-- ── HR Zones ─────────────────────── -->
  <div class="section">
    <div class="section-title">Heart Rate Zones</div>
    <div class="zone-legend">
      <span><span class="zone-dot" style="background:var(--zone1)"></span>Z1 Recovery</span>
      <span><span class="zone-dot" style="background:var(--zone2)"></span>Z2 Easy</span>
      <span><span class="zone-dot" style="background:var(--zone3)"></span>Z3 Aerobic</span>
      <span><span class="zone-dot" style="background:var(--zone4)"></span>Z4 Threshold</span>
      <span><span class="zone-dot" style="background:var(--zone5)"></span>Z5 Max</span>
    </div>
    <div class="charts-row">
      <div class="chart-card">
        <h3>Zone Distribution (All Runs)</h3>
        <canvas id="zoneDonut" height="260"></canvas>
      </div>
      <div class="chart-card">
        <h3>Zone Distribution Over Time</h3>
        <canvas id="zoneTimeline" height="260"></canvas>
      </div>
    </div>
  </div>

  <!-- ── Cardiac Efficiency ───────────── -->
  <div class="section">
    <div class="section-title">Cardiac Efficiency</div>
    <div class="charts-row">
      <div class="chart-card">
        <h3>Pace vs Heart Rate</h3>
        <canvas id="paceHrScatter" height="280"></canvas>
      </div>
      <div class="chart-card">
        <h3>Fitness Proxy: HR at 6-7 min/km Pace</h3>
        <canvas id="fitnessTrend" height="280"></canvas>
      </div>
    </div>
  </div>

  <!-- ── Individual Run Viewer ────────── -->
  <div class="section">
    <div class="section-title">Run Viewer</div>
    <select class="run-dropdown" id="runSelector">
      <!-- populated by JS -->
    </select>
    <div class="charts-row">
      <div class="chart-card full">
        <h3 id="runViewerTitle">Select a run</h3>
        <canvas id="runViewer" height="300"></canvas>
      </div>
    </div>
  </div>

  <!-- ── Splits ───────────────────────── -->
  <div class="section">
    <div class="section-title">Splits Analysis</div>
    <div class="chart-card full">
      <h3 id="splitsTitle">Splits</h3>
      <div id="splitsContainer">
        <table class="splits-table">
          <thead><tr><th>KM</th><th>Pace</th><th>Avg HR</th><th></th></tr></thead>
          <tbody id="splitsBody"></tbody>
        </table>
      </div>
    </div>
  </div>

</div>

<script>
// ── Embedded Data ─────────────────────────────────────────────────────────
const RUNS = {json.dumps(runs_data, ensure_ascii=False)};
const PANEL = {json.dumps(panel_records, ensure_ascii=False)};
const MONTHLY = {json.dumps(monthly_records, ensure_ascii=False)};
const ZONES_PER_RUN = {json.dumps(zone_data_per_run, ensure_ascii=False)};
const MAX_HR = {max_hr_observed};
const PLAN = {json.dumps(plan_json, ensure_ascii=False)};
const ADHERENCE = {json.dumps(adherence, ensure_ascii=False)};
const CURRENT_WEEK = {cw_num};

// ── Chart.js Defaults ────────────────────────────────────────────────────
Chart.defaults.color = '#7a7880';
Chart.defaults.borderColor = '#222230';
Chart.defaults.font.family = "'Space Mono', monospace";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 8;
Chart.defaults.plugins.tooltip.backgroundColor = '#1a1a22';
Chart.defaults.plugins.tooltip.borderColor = '#333';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.cornerRadius = 6;
Chart.defaults.plugins.tooltip.padding = 10;

const ZONE_COLORS = ['#3b82f6','#22c55e','#f5a623','#ff6b4a','#ff4a7a'];
const ZONE_NAMES = ['Z1','Z2','Z3','Z4','Z5'];

// ── Training Plan: Shared Config ─────────────────────────────────────────
const TP_COLORS = {{
  'Foundation': '#4ad4e0',
  'Building': '#f5a623',
  'Development': '#9b7aff',
}};
const TP_TYPE_COLORS = {{
  'cycling_z2': '#4ad4e0',
  'run_walk': '#f5a623',
  'easy_run': '#22c55e',
  'tempo_run': '#ff4a7a',
  'rest': '#333',
}};
const TP_TYPE_LABELS = {{
  'cycling_z2': 'Cycling Zone 2',
  'run_walk': 'Run / Walk',
  'easy_run': 'Easy Run',
  'tempo_run': 'Tempo Run',
  'rest': 'Rest',
}};

// ── Training Plan: Hero Ring + Current Week ──────────────────────────────
(() => {{
  const today = new Date().toISOString().slice(0, 10);

  // Ring
  const ringEl = document.getElementById('tpRing');
  if (ringEl) {{
    const pct = CURRENT_WEEK > 0 ? (CURRENT_WEEK / 26) * 100 : 0;
    const circ = 2 * Math.PI * 50;
    const off = circ - (pct / 100) * circ;
    const week = PLAN.find(w => w.week === CURRENT_WEEK);
    const col = week ? TP_COLORS[week.phase] : '#333';
    ringEl.innerHTML = `
      <svg width="120" height="120" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="50" fill="none" stroke="#1a1a22" stroke-width="6"/>
        <circle cx="60" cy="60" r="50" fill="none" stroke="${{col}}" stroke-width="6"
          stroke-dasharray="${{circ}}" stroke-dashoffset="${{off}}" stroke-linecap="round"/>
      </svg>
      <div class="tp-ring-label">
        <span class="tp-ring-week">${{CURRENT_WEEK || '\u2014'}}</span>
        <span class="tp-ring-sub">of 26</span>
      </div>`;
  }}

  // Phase name
  const phaseEl = document.getElementById('tpPhaseName');
  if (phaseEl && CURRENT_WEEK > 0) {{
    const week = PLAN.find(w => w.week === CURRENT_WEEK);
    if (week) {{
      phaseEl.textContent = week.phase;
      phaseEl.className = 'tp-phase-name tp-phase-' + week.phase.toLowerCase();
    }}
  }}

  // Current week strip
  const strip = document.getElementById('tpWeekStrip');
  if (!CURRENT_WEEK || CURRENT_WEEK === 0) {{
    strip.innerHTML = '<div style="grid-column:1/-1;color:var(--text-muted);font-family:var(--font-mono);font-size:13px;padding:24px">Plan starts {PLAN_START.isoformat()}</div>';
    return;
  }}
  const week = PLAN.find(w => w.week === CURRENT_WEEK);
  if (!week) return;

  week.sessions.forEach(s => {{
    const isToday = s.date === today;
    const isRest = s.type === 'rest';
    const col = TP_TYPE_COLORS[s.type];
    const card = document.createElement('div');
    card.className = 'tp-sc' + (isToday ? ' is-today' : '') + (isRest ? ' is-rest' : '');
    card.innerHTML = `
      <div class="tp-sc-bar" style="background:${{col}}"></div>
      <div class="tp-sc-day">${{s.day.slice(0,3)}}</div>
      <div class="tp-sc-type">${{TP_TYPE_LABELS[s.type]}}</div>
      ${{isRest ? '' : '<div class="tp-sc-dur">' + s.duration_min + ' min</div>'}}
      ${{isRest ? '' : '<div class="tp-sc-hr">HR ' + s.hr_target[0] + '-' + s.hr_target[1] + '</div>'}}`;
    strip.appendChild(card);
  }});
}})();

// ── Training Plan: 26-Week Timeline ──────────────────────────────────────
(() => {{
  const container = document.getElementById('tpTimelineRow');
  const today = new Date().toISOString().slice(0, 10);

  PLAN.forEach(w => {{
    const cell = document.createElement('div');
    cell.className = 'tp-tl-cell';
    const isCurrent = w.week === CURRENT_WEEK;
    const isFuture = w.start_date > today;
    const col = TP_COLORS[w.phase];

    if (isCurrent) {{
      cell.classList.add('current');
      cell.style.background = col;
    }} else if (isFuture) {{
      cell.classList.add('future');
    }} else {{
      cell.classList.add('past');
      const adh = ADHERENCE.find(a => a.week === w.week);
      const pct = adh ? adh.completion_pct : 0;
      cell.style.background = `linear-gradient(to top, ${{col}}80 ${{pct}}%, transparent ${{pct}}%)`;
    }}
    cell.textContent = w.week;
    cell.title = `Week ${{w.week}} (${{w.phase}}): ${{w.start_date}} \u2014 ${{w.end_date}}`;
    container.appendChild(cell);
  }});
}})();

// ── Training Plan: Full Plan Accordion ───────────────────────────────────
(() => {{
  const container = document.getElementById('tpFullPlan');
  const today = new Date().toISOString().slice(0, 10);

  const phases = [
    {{ name: 'Foundation', weeks: PLAN.filter(w => w.phase === 'Foundation'), range: 'Weeks 1 \u2013 8' }},
    {{ name: 'Building', weeks: PLAN.filter(w => w.phase === 'Building'), range: 'Weeks 9 \u2013 17' }},
    {{ name: 'Development', weeks: PLAN.filter(w => w.phase === 'Development'), range: 'Weeks 18 \u2013 26' }},
  ];

  phases.forEach((phase, pi) => {{
    const group = document.createElement('div');
    group.className = 'tp-pg';
    const col = TP_COLORS[phase.name];
    const hasCurrentWeek = phase.weeks.some(w => w.week === CURRENT_WEEK);
    const isOpen = hasCurrentWeek || pi === 0;

    const hd = document.createElement('div');
    hd.className = 'tp-pg-hd' + (isOpen ? ' open' : ' collapsed');
    hd.innerHTML = `
      <div class="tp-pg-bar" style="background:${{col}}"></div>
      <div class="tp-pg-title" style="color:${{col}}">${{phase.name}}</div>
      <div class="tp-pg-range">${{phase.range}}</div>
      <div class="tp-pg-chev">\u25BC</div>`;

    const body = document.createElement('div');
    body.className = 'tp-pg-body' + (isOpen ? ' open' : '');

    hd.addEventListener('click', () => {{
      hd.classList.toggle('open');
      hd.classList.toggle('collapsed');
      body.classList.toggle('open');
    }});

    phase.weeks.forEach(w => {{
      const adh = ADHERENCE.find(a => a.week === w.week);
      const pct = adh ? adh.completion_pct : 0;
      const isCurrent = w.week === CURRENT_WEEK;
      const isFuture = w.start_date > today;

      const wa = document.createElement('div');
      wa.className = 'tp-wa';

      const wHd = document.createElement('div');
      wHd.className = 'tp-wa-hd' + (isCurrent ? ' cw' : '');
      wHd.innerHTML = `
        <div class="tp-wa-num">W${{w.week}}</div>
        <div class="tp-wa-dates">${{w.start_date.slice(5)}} \u2192 ${{w.end_date.slice(5)}}</div>
        <div class="tp-wa-prog"><div class="tp-wa-prog-fill" style="width:${{pct}}%;background:${{col}}"></div></div>
        <div class="tp-wa-pct">${{isFuture ? '\u2014' : pct + '%'}}</div>
        <div class="tp-wa-chev">\u25BC</div>`;

      const wDet = document.createElement('div');
      wDet.className = 'tp-wa-det';

      const sg = document.createElement('div');
      sg.className = 'tp-sg';
      w.sessions.forEach(s => {{
        const isRest = s.type === 'rest';
        const row = document.createElement('div');
        row.className = 'tp-sr';
        row.innerHTML = `
          <div class="tp-sr-day">${{s.day.slice(0,3)}}</div>
          <div class="tp-sr-type"><span class="tp-sr-dot" style="background:${{TP_TYPE_COLORS[s.type]}}"></span><span class="tp-sr-name">${{TP_TYPE_LABELS[s.type]}}</span></div>
          <div class="tp-sr-dur">${{isRest ? '\u2014' : s.duration_min + ' min'}}</div>
          <div class="tp-sr-hr">${{isRest ? '' : 'HR ' + s.hr_target[0] + '-' + s.hr_target[1]}}</div>`;
        sg.appendChild(row);
      }});
      wDet.appendChild(sg);

      wHd.addEventListener('click', () => {{
        wHd.classList.toggle('open');
        wDet.classList.toggle('open');
      }});

      if (isCurrent) {{
        wHd.classList.add('open');
        wDet.classList.add('open');
      }}

      wa.appendChild(wHd);
      wa.appendChild(wDet);
      body.appendChild(wa);
    }});

    group.appendChild(hd);
    group.appendChild(body);
    container.appendChild(group);
  }});
}})();

// ── Training Plan: Adherence Chart ───────────────────────────────────────
(() => {{
  const past = ADHERENCE.filter(a => a.planned > 0);
  if (!past.length) return;
  const ctx = document.getElementById('adherenceChart').getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: past.map(a => 'W' + a.week),
      datasets: [
        {{
          label: 'Completion %',
          data: past.map(a => a.completion_pct),
          backgroundColor: 'rgba(34,197,94,0.7)',
          borderRadius: 4,
        }},
        {{
          label: 'HR Compliance %',
          data: past.map(a => a.hr_compliance_pct),
          backgroundColor: 'rgba(74,212,224,0.7)',
          borderRadius: 4,
        }}
      ]
    }},
    options: {{
      responsive: true,
      scales: {{
        y: {{ max: 100, title: {{ display: true, text: '%' }}, grid: {{ color: '#1a1a22' }} }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});
}})();

// ── Training Plan: HR vs Target Scatter ──────────────────────────────────
(() => {{
  const ctx = document.getElementById('hrTargetChart').getContext('2d');
  const planStart = PLAN[0].start_date;
  const planEnd = PLAN[PLAN.length - 1].end_date;
  const planRuns = PANEL.filter(r =>
    r.date >= planStart && r.date <= planEnd && r.average_heartrate
  );

  new Chart(ctx, {{
    type: 'scatter',
    data: {{
      datasets: [{{
        label: 'Actual HR',
        data: planRuns.map(r => ({{ x: r.date, y: r.average_heartrate }})),
        pointBackgroundColor: planRuns.map(r => {{
          const hr = r.average_heartrate;
          if (hr >= {TARGET_LO} && hr <= {TARGET_HI}) return '#22c55e';
          if (hr < {TARGET_LO}) return '#4ad4e0';
          return '#ff4a7a';
        }}),
        pointRadius: 6,
        pointBorderColor: 'transparent',
      }}]
    }},
    options: {{
      responsive: true,
      scales: {{
        x: {{ type: 'category', grid: {{ display: false }}, ticks: {{ maxTicksLimit: 8 }} }},
        y: {{
          title: {{ display: true, text: 'HR (bpm)' }},
          grid: {{ color: '#1a1a22' }},
        }}
      }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => {{
              const r = planRuns[ctx.dataIndex];
              return r.date.slice(0,10) + ' \u2014 ' + r.average_heartrate + ' bpm';
            }}
          }}
        }}
      }}
    }}
  }});
}})();

// ── Monthly Volume Chart ─────────────────────────────────────────────────
(() => {{
  const ctx = document.getElementById('monthlyVolume').getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: MONTHLY.map(m => m.month_label),
      datasets: [
        {{
          label: 'Distance (km)',
          data: MONTHLY.map(m => m.total_km ? +m.total_km.toFixed(1) : 0),
          backgroundColor: 'rgba(255,107,74,0.7)',
          borderRadius: 4,
          yAxisID: 'y',
        }},
        {{
          label: 'Runs',
          data: MONTHLY.map(m => m.runs),
          type: 'line',
          borderColor: '#4ad4e0',
          backgroundColor: 'rgba(74,212,224,0.1)',
          pointRadius: 4,
          pointBackgroundColor: '#4ad4e0',
          tension: 0.3,
          yAxisID: 'y1',
        }}
      ]
    }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'index', intersect: false }},
      scales: {{
        y: {{ position: 'left', title: {{ display: true, text: 'km' }}, grid: {{ color: '#1a1a22' }} }},
        y1: {{ position: 'right', title: {{ display: true, text: 'runs' }}, grid: {{ drawOnChartArea: false }} }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});
}})();

// ── Monthly Trend Chart ──────────────────────────────────────────────────
(() => {{
  const ctx = document.getElementById('monthlyTrend').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: MONTHLY.map(m => m.month_label),
      datasets: [
        {{
          label: 'Avg HR (bpm)',
          data: MONTHLY.map(m => m.avg_hr ? +m.avg_hr.toFixed(0) : null),
          borderColor: '#ff4a7a',
          backgroundColor: 'rgba(255,74,122,0.1)',
          pointRadius: 5,
          pointBackgroundColor: '#ff4a7a',
          tension: 0.3,
          yAxisID: 'y',
        }},
        {{
          label: 'Avg Pace (min/km)',
          data: MONTHLY.map(m => m.avg_pace && m.avg_pace < 20 ? +m.avg_pace.toFixed(1) : null),
          borderColor: '#9b7aff',
          backgroundColor: 'rgba(155,122,255,0.1)',
          pointRadius: 5,
          pointBackgroundColor: '#9b7aff',
          tension: 0.3,
          yAxisID: 'y1',
        }}
      ]
    }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'index', intersect: false }},
      scales: {{
        y: {{ position: 'left', title: {{ display: true, text: 'bpm' }}, grid: {{ color: '#1a1a22' }} }},
        y1: {{ position: 'right', title: {{ display: true, text: 'min/km' }}, grid: {{ drawOnChartArea: false }}, reverse: true }},
        x: {{ grid: {{ display: false }} }}
      }}
    }}
  }});
}})();

// ── Zone Donut ───────────────────────────────────────────────────────────
(() => {{
  const totalZones = [0,0,0,0,0];
  ZONES_PER_RUN.forEach(r => {{
    r.zones_sec.forEach((s, i) => totalZones[i] += s);
  }});
  const total = totalZones.reduce((a, b) => a + b, 0);
  const ctx = document.getElementById('zoneDonut').getContext('2d');
  new Chart(ctx, {{
    type: 'doughnut',
    data: {{
      labels: ZONE_NAMES.map((z, i) => z + ' (' + (100 * totalZones[i] / total).toFixed(1) + '%)'),
      datasets: [{{
        data: totalZones,
        backgroundColor: ZONE_COLORS,
        borderColor: '#111115',
        borderWidth: 3,
      }}]
    }},
    options: {{
      responsive: true,
      cutout: '55%',
      plugins: {{
        legend: {{ position: 'right' }}
      }}
    }}
  }});
}})();

// ── Zone Timeline (stacked bar) ──────────────────────────────────────────
(() => {{
  const ctx = document.getElementById('zoneTimeline').getContext('2d');
  const datasets = ZONE_NAMES.map((name, i) => ({{
    label: name,
    data: ZONES_PER_RUN.map(r => r.zones_pct[i]),
    backgroundColor: ZONE_COLORS[i],
    borderRadius: 2,
  }}));
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: ZONES_PER_RUN.map(r => r.date),
      datasets
    }},
    options: {{
      responsive: true,
      scales: {{
        x: {{ stacked: true, grid: {{ display: false }}, ticks: {{ maxTicksLimit: 12 }} }},
        y: {{ stacked: true, max: 100, title: {{ display: true, text: '%' }}, grid: {{ color: '#1a1a22' }} }}
      }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => ctx.dataset.label + ': ' + ctx.raw.toFixed(1) + '%'
          }}
        }}
      }}
    }}
  }});
}})();

// ── Pace vs HR Scatter ───────────────────────────────────────────────────
(() => {{
  const validRuns = PANEL.filter(r => r.average_heartrate && r.pace_min_per_km && r.pace_min_per_km < 20);
  const ctx = document.getElementById('paceHrScatter').getContext('2d');

  // Color by date (older = dim, newer = bright)
  const dates = validRuns.map(r => new Date(r.date).getTime());
  const minD = Math.min(...dates);
  const maxD = Math.max(...dates);
  const range = maxD - minD || 1;

  new Chart(ctx, {{
    type: 'scatter',
    data: {{
      datasets: [{{
        label: 'Runs',
        data: validRuns.map(r => ({{ x: r.pace_min_per_km, y: r.average_heartrate }})),
        pointBackgroundColor: validRuns.map(r => {{
          const t = (new Date(r.date).getTime() - minD) / range;
          return `rgba(255,${{Math.round(107 - t * 60)}},${{Math.round(74 + t * 100)}},${{0.4 + t * 0.6}})`;
        }}),
        pointRadius: 7,
        pointHoverRadius: 10,
        pointBorderColor: 'transparent',
      }}]
    }},
    options: {{
      responsive: true,
      scales: {{
        x: {{ title: {{ display: true, text: 'Pace (min/km)' }}, grid: {{ color: '#1a1a22' }} }},
        y: {{ title: {{ display: true, text: 'Avg HR (bpm)' }}, grid: {{ color: '#1a1a22' }} }}
      }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => {{
              const r = validRuns[ctx.dataIndex];
              return r.date.slice(0,10) + ' — ' + r.pace_min_per_km + ' min/km, ' + r.average_heartrate + ' bpm';
            }}
          }}
        }}
      }}
    }}
  }});
}})();

// ── Fitness Trend (HR at similar pace over time) ─────────────────────────
(() => {{
  // Filter runs with pace between 6-7 min/km (moderate effort)
  const similar = PANEL
    .filter(r => r.average_heartrate && r.pace_min_per_km >= 5 && r.pace_min_per_km <= 8)
    .sort((a, b) => a.date.localeCompare(b.date));

  const ctx = document.getElementById('fitnessTrend').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: similar.map(r => r.date.slice(0, 10)),
      datasets: [
        {{
          label: 'Avg HR at 5-8 min/km pace',
          data: similar.map(r => r.average_heartrate),
          borderColor: '#ff6b4a',
          backgroundColor: 'rgba(255,107,74,0.08)',
          fill: true,
          pointRadius: 5,
          pointBackgroundColor: '#ff6b4a',
          tension: 0.3,
        }},
        {{
          label: 'Pace (min/km)',
          data: similar.map(r => r.pace_min_per_km),
          borderColor: '#4ad4e0',
          pointRadius: 4,
          pointBackgroundColor: '#4ad4e0',
          borderDash: [4, 4],
          tension: 0.3,
          yAxisID: 'y1',
        }}
      ]
    }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'index', intersect: false }},
      scales: {{
        x: {{ grid: {{ display: false }}, ticks: {{ maxTicksLimit: 10 }} }},
        y: {{ position: 'left', title: {{ display: true, text: 'HR (bpm)' }}, grid: {{ color: '#1a1a22' }} }},
        y1: {{ position: 'right', title: {{ display: true, text: 'min/km' }}, grid: {{ drawOnChartArea: false }}, reverse: true }}
      }},
      plugins: {{
        tooltip: {{
          callbacks: {{
            afterBody: (items) => {{
              const r = similar[items[0].dataIndex];
              return r.name + ' — ' + (r.distance_km || 0).toFixed(1) + ' km';
            }}
          }}
        }}
      }}
    }}
  }});
}})();

// ── Run Viewer ───────────────────────────────────────────────────────────
const runSelector = document.getElementById('runSelector');
let runViewerChart = null;

// Populate dropdown (newest first)
const sortedRuns = [...RUNS].reverse();
sortedRuns.forEach((r, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = r.date + ' — ' + r.name + ' (' + r.distance_km + ' km)';
  runSelector.appendChild(opt);
}});

function renderRunViewer(run) {{
  const ctx = document.getElementById('runViewer').getContext('2d');
  document.getElementById('runViewerTitle').textContent = run.name + ' — ' + run.date + ' — ' + run.distance_km + ' km';

  if (runViewerChart) runViewerChart.destroy();

  const timeLabels = run.time.map(t => {{
    const m = Math.floor(t / 60);
    const s = t % 60;
    return m + ':' + String(s).padStart(2, '0');
  }});

  const datasets = [];

  if (run.heartrate.length) {{
    datasets.push({{
      label: 'Heart Rate (bpm)',
      data: run.heartrate,
      borderColor: '#ff4a7a',
      backgroundColor: 'rgba(255,74,122,0.06)',
      fill: true,
      pointRadius: 0,
      borderWidth: 1.5,
      tension: 0.2,
      yAxisID: 'y',
    }});
  }}

  if (run.altitude.length) {{
    datasets.push({{
      label: 'Elevation (m)',
      data: run.altitude,
      borderColor: '#4ad4e0',
      backgroundColor: 'rgba(74,212,224,0.08)',
      fill: true,
      pointRadius: 0,
      borderWidth: 1.5,
      tension: 0.2,
      yAxisID: 'y1',
    }});
  }}

  if (run.velocity.length) {{
    datasets.push({{
      label: 'Speed (m/s)',
      data: run.velocity,
      borderColor: 'rgba(155,122,255,0.5)',
      pointRadius: 0,
      borderWidth: 1,
      tension: 0.2,
      yAxisID: 'y2',
    }});
  }}

  runViewerChart = new Chart(ctx, {{
    type: 'line',
    data: {{ labels: timeLabels, datasets }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'index', intersect: false }},
      scales: {{
        x: {{ grid: {{ display: false }}, ticks: {{ maxTicksLimit: 15 }} }},
        y: {{
          position: 'left',
          title: {{ display: true, text: 'HR (bpm)' }},
          grid: {{ color: '#1a1a22' }},
        }},
        y1: {{
          position: 'right',
          title: {{ display: true, text: 'Elev (m)' }},
          grid: {{ drawOnChartArea: false }},
        }},
        y2: {{
          display: false,
        }}
      }},
      plugins: {{
        legend: {{ position: 'top' }}
      }}
    }}
  }});

  // Render splits
  renderSplits(run);
}}

function renderSplits(run) {{
  const tbody = document.getElementById('splitsBody');
  const title = document.getElementById('splitsTitle');
  tbody.innerHTML = '';

  if (!run.splits || !run.splits.length) {{
    title.textContent = 'Splits — no km splits for this run';
    return;
  }}

  title.textContent = 'Splits — ' + run.name + ' (' + run.splits.length + ' km)';
  const maxPace = Math.max(...run.splits.map(s => s.pace_min_km));

  run.splits.forEach(s => {{
    const tr = document.createElement('tr');
    const paceMin = Math.floor(s.pace_min_km);
    const paceSec = Math.round((s.pace_min_km - paceMin) * 60);
    const barW = Math.round(100 * s.pace_min_km / maxPace);
    tr.innerHTML = `
      <td>${{s.km}}</td>
      <td>${{paceMin}}:${{String(paceSec).padStart(2,'0')}} /km</td>
      <td>${{s.avg_hr ? s.avg_hr + ' bpm' : '—'}}</td>
      <td><span class="pace-bar" style="width:${{barW}}px"></span></td>
    `;
    tbody.appendChild(tr);
  }});
}}

// Load first run
runSelector.addEventListener('change', (e) => {{
  renderRunViewer(sortedRuns[+e.target.value]);
}});
if (sortedRuns.length) renderRunViewer(sortedRuns[0]);

</script>
</body>
</html>"""

    with open(DASHBOARD_FILE, "w") as f:
        f.write(html)

    print(f"Dashboard written to {DASHBOARD_FILE} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    build()
