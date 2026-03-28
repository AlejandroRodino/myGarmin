"""
Fetch all run and ride activities from Strava and build a panel dataset.

Panel structure:
  - Unit:      runner (you)
  - Time:      each activity observation (date)
  - Variables: distance, duration, pace, heart rate, cadence, elevation, etc.

Output: runs_panel.csv
"""

import os
import time

import pandas as pd
import requests

from strava_auth import get_access_token

API_BASE = "https://www.strava.com/api/v3"
BASE_DIR = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(BASE_DIR, "runs_panel.csv")


def fetch_all_activities(access_token, types=("Run", "Ride")):
    """Paginate through all activities and keep runs and rides."""
    headers = {"Authorization": f"Bearer {access_token}"}
    all_activities = []
    page = 1
    per_page = 200  # max allowed

    while True:
        print(f"Fetching page {page}...")
        resp = requests.get(
            f"{API_BASE}/athlete/activities",
            headers=headers,
            params={"page": page, "per_page": per_page},
        )
        resp.raise_for_status()
        activities = resp.json()

        if not activities:
            break

        matched = [a for a in activities if a["type"] in types]
        all_activities.extend(matched)
        print(f"  Found {len(matched)} activities (total so far: {len(all_activities)})")

        if len(activities) < per_page:
            break

        page += 1
        time.sleep(0.5)  # respect rate limits

    return all_activities


def build_panel(activities):
    """Transform raw Strava activities into a clean panel DataFrame."""
    records = []
    for r in activities:
        distance_km = r["distance"] / 1000
        duration_min = r["moving_time"] / 60
        pace_min_km = duration_min / distance_km if distance_km > 0 else None

        records.append({
            # Identifiers
            "activity_id": r["id"],
            "runner": "self",
            "activity_type": r["type"],

            # Time dimension
            "date": r["start_date_local"][:10],
            "start_time": r["start_date_local"],
            "year": int(r["start_date_local"][:4]),
            "month": int(r["start_date_local"][5:7]),
            "day_of_week": pd.Timestamp(r["start_date_local"]).day_name(),

            # Core metrics
            "distance_km": round(distance_km, 2),
            "moving_time_min": round(duration_min, 2),
            "elapsed_time_min": round(r["elapsed_time"] / 60, 2),
            "pace_min_per_km": round(pace_min_km, 2) if pace_min_km else None,

            # Physiological
            "average_heartrate": r.get("average_heartrate"),
            "max_heartrate": r.get("max_heartrate"),
            "average_cadence": r.get("average_cadence"),

            # Terrain
            "total_elevation_gain_m": r.get("total_elevation_gain"),
            "elev_high_m": r.get("elev_high"),
            "elev_low_m": r.get("elev_low"),

            # Performance
            "average_speed_kmh": round(r["average_speed"] * 3.6, 2),
            "max_speed_kmh": round(r["max_speed"] * 3.6, 2),
            "suffer_score": r.get("suffer_score"),

            # Context
            "name": r.get("name"),
            "workout_type": r.get("workout_type"),
            "gear_id": r.get("gear_id"),
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Panel index: runner x time
    df["run_number"] = range(1, len(df) + 1)

    return df


def fetch_and_save():
    """Fetch activities and save panel CSV. Importable by app.py."""
    token = get_access_token()
    print("\nFetching activities from Strava...\n")
    activities = fetch_all_activities(token)
    print(f"\nTotal activities fetched: {len(activities)}")

    if not activities:
        print("No activities found.")
        return

    df = build_panel(activities)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPanel dataset saved to {OUTPUT_PATH}")
    print(f"Shape: {df.shape[0]} observations x {df.shape[1]} variables")
    return df


def main():
    df = fetch_and_save()
    if df is not None:
        print(f"\nDate range: {df['date'].min().date()} -> {df['date'].max().date()}")
        print(f"\nSummary stats:")
        print(df[["distance_km", "moving_time_min", "pace_min_per_km",
                  "average_heartrate", "total_elevation_gain_m"]].describe().round(2))


if __name__ == "__main__":
    main()
