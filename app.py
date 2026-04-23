"""
Flask server for Strava dashboard + fitness-trend page with daily auto-refresh.
Designed for Railway deployment.
"""

import os
import traceback

from flask import Flask, send_file, jsonify, render_template, abort, redirect, url_for

from apscheduler.schedulers.background import BackgroundScheduler

from strava_auth import bootstrap_tokens_from_env

BASE_DIR = os.path.dirname(__file__)
DASHBOARD_FILE = os.path.join(BASE_DIR, "dashboard.html")
FITNESS_PLOT = os.path.join(BASE_DIR, "fitness_trend.png")
PANEL_FILE = os.path.join(BASE_DIR, "runs_panel.csv")

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


def run_fitness_update():
    """Lightweight refresh: just activities + regression. No streams, no dashboard."""
    bootstrap_tokens_from_env()

    from fetch_runs import fetch_and_save
    print("[fitness] Fetching activities...")
    fetch_and_save()

    from fitness_trend import run_analysis
    print("[fitness] Running regression...")
    return run_analysis()


@app.route("/")
def index():
    if not os.path.exists(DASHBOARD_FILE):
        return redirect(url_for("fitness_page"))
    return send_file(DASHBOARD_FILE)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "dashboard_exists": os.path.exists(DASHBOARD_FILE),
        "panel_exists": os.path.exists(PANEL_FILE),
        "fitness_plot_exists": os.path.exists(FITNESS_PLOT),
    })


@app.route("/refresh", methods=["POST"])
def refresh():
    run_pipeline()
    return jsonify({"status": "refreshed"})


@app.route("/fitness")
def fitness_page():
    return render_template("fitness.html")


@app.route("/fitness/plot")
def fitness_plot():
    if not os.path.exists(FITNESS_PLOT):
        abort(404)
    # Disable caching so the browser sees the freshly-written PNG after each update
    resp = send_file(FITNESS_PLOT, mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@app.route("/fitness/update", methods=["POST"])
def fitness_update():
    try:
        result = run_fitness_update()
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    bootstrap_tokens_from_env()

    if not os.path.exists(DASHBOARD_FILE):
        print("[startup] No dashboard found, running initial pipeline...")
        run_pipeline()

    scheduler = BackgroundScheduler()
    scheduler.add_job(run_pipeline, "cron", hour=6, minute=0)
    scheduler.start()
    print("[scheduler] Daily refresh scheduled at 06:00 UTC")

    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
