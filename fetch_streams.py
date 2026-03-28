"""
Fetch second-by-second stream data for all runs from Strava.
Streams: heartrate, velocity, altitude, cadence, distance, time.
Saves to streams_data.json
"""

import json
import os
import time

import pandas as pd
import requests

from strava_auth import get_access_token

API_BASE = "https://www.strava.com/api/v3"
STREAMS_FILE = os.path.join(os.path.dirname(__file__), "streams_data.json")


def fetch_stream(activity_id, access_token):
    """Fetch all available streams for a single activity."""
    headers = {"Authorization": f"Bearer {access_token}"}
    stream_types = "time,distance,altitude,heartrate,cadence,velocity_smooth,grade_smooth"

    resp = requests.get(
        f"{API_BASE}/activities/{activity_id}/streams",
        headers=headers,
        params={"keys": stream_types, "key_type": "time"},
    )

    if resp.status_code == 429:
        # Rate limited — wait and retry
        print("    Rate limited, waiting 60s...")
        time.sleep(60)
        return fetch_stream(activity_id, access_token)

    if resp.status_code != 200:
        print(f"    Error {resp.status_code} for activity {activity_id}")
        return None

    streams = resp.json()
    return {s["type"]: s["data"] for s in streams}


PANEL_FILE = os.path.join(os.path.dirname(__file__), "runs_panel.csv")


def main():
    token = get_access_token()

    # Load panel to get activity IDs and metadata
    df = pd.read_csv(PANEL_FILE, parse_dates=["date"])

    # Check if we have partial progress
    existing = {}
    if os.path.exists(STREAMS_FILE):
        with open(STREAMS_FILE) as f:
            existing = json.load(f)
        print(f"Resuming — {len(existing)} activities already fetched.")

    all_streams = existing.copy()
    activities = df[["activity_id", "date", "name", "distance_km"]].to_dict("records")

    for i, act in enumerate(activities):
        aid = str(act["activity_id"])
        if aid in all_streams:
            continue

        print(f"[{i+1}/{len(activities)}] {str(act['date'])[:10]} — {act['name']} ({act['distance_km']} km)")
        streams = fetch_stream(act["activity_id"], token)

        if streams:
            all_streams[aid] = {
                "activity_id": act["activity_id"],
                "date": str(act["date"])[:10],
                "name": act["name"],
                "distance_km": act["distance_km"],
                "streams": streams,
            }
            print(f"    ✓ {len(streams.get('time', []))} data points")
        else:
            print(f"    ✗ No stream data")

        # Save progress after each fetch
        with open(STREAMS_FILE, "w") as f:
            json.dump(all_streams, f)

        time.sleep(1.0)  # respect rate limits (600 req / 15 min)

    print(f"\nDone. {len(all_streams)} activities with stream data saved to {STREAMS_FILE}")


def fetch_and_save():
    """Fetch streams and save to JSON. Importable by app.py."""
    main()


if __name__ == "__main__":
    main()
