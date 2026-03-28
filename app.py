"""
Flask server for Strava dashboard with daily auto-refresh.
Designed for Railway deployment.
"""

import os
import traceback

from flask import Flask, send_file, jsonify

from apscheduler.schedulers.background import BackgroundScheduler

from strava_auth import bootstrap_tokens_from_env

BASE_DIR = os.path.dirname(__file__)
DASHBOARD_FILE = os.path.join(BASE_DIR, "dashboard.html")

app = Flask(__name__)


def run_pipeline():
    """Execute the full data pipeline: fetch → streams → dashboard."""
    try:
        print("[pipeline] Starting data refresh...")
        bootstrap_tokens_from_env()

        from fetch_runs import fetch_and_save
        print("[pipeline] Fetching activities...")
        fetch_and_save()

        from fetch_streams import fetch_and_save as fetch_streams
        print("[pipeline] Fetching streams...")
        fetch_streams()

        from build_dashboard import build
        print("[pipeline] Building dashboard...")
        build()

        print("[pipeline] Done.")
    except Exception:
        traceback.print_exc()
        print("[pipeline] Failed — dashboard may be stale.")


@app.route("/")
def index():
    if not os.path.exists(DASHBOARD_FILE):
        return "<h2>Dashboard not built yet. Trigger /refresh or wait for scheduled build.</h2>", 503
    return send_file(DASHBOARD_FILE)


@app.route("/health")
def health():
    has_dashboard = os.path.exists(DASHBOARD_FILE)
    return jsonify({"status": "ok", "dashboard_exists": has_dashboard})


@app.route("/refresh", methods=["POST"])
def refresh():
    run_pipeline()
    return jsonify({"status": "refreshed"})


if __name__ == "__main__":
    # Bootstrap tokens from env if available (Railway)
    bootstrap_tokens_from_env()

    # Build dashboard on startup if it doesn't exist
    if not os.path.exists(DASHBOARD_FILE):
        print("[startup] No dashboard found, running initial pipeline...")
        run_pipeline()

    # Schedule daily refresh at 06:00 UTC
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_pipeline, "cron", hour=6, minute=0)
    scheduler.start()
    print("[scheduler] Daily refresh scheduled at 06:00 UTC")

    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
