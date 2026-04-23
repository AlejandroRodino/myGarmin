"""
Estimate HR ~ const + pace + elevation on Runs, then treat each run's
residual as its fixed effect (ε̂_i) and plot over time.

With one observation per run, a per-run dummy is not separately identified
from pace/elevation — the residual is the fixed-effect estimator's
equivalent (the run's deviation from the conditional mean).
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

BASE_DIR = os.path.dirname(__file__)
PANEL = os.path.join(BASE_DIR, "runs_panel.csv")
OUT_PLOT = os.path.join(BASE_DIR, "run_fixed_effects.png")


def main():
    df = pd.read_csv(PANEL, parse_dates=["date"])
    df = df[df["activity_type"] == "Run"].copy()
    df = df.dropna(subset=["average_heartrate", "pace_min_per_km", "total_elevation_gain_m"])
    n_raw = len(df)

    # Outlier filter: drop sub-500m "runs" and implausible paces.
    # Sub-500m activities are usually stops/tests with unreliable pace.
    # Human running pace plausible range ≈ 3-10 min/km.
    df = df[df["distance_km"] >= 0.5]
    df = df[df["pace_min_per_km"].between(3.0, 10.0)]
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Runs: {n_raw} raw → {len(df)} after outlier filter "
          f"(dropped dist<0.5km or pace∉[3,10] min/km)")

    X = df[["pace_min_per_km", "total_elevation_gain_m"]]
    X = sm.add_constant(X)
    y = df["average_heartrate"]

    model = sm.OLS(y, X).fit(cov_type="HC1")
    print(model.summary())

    df["fitted_hr"] = model.fittedvalues
    df["run_fe"] = model.resid  # run-level fixed effect (= residual)

    # Rolling mean for readability (window = 8 runs)
    df["fe_roll"] = df["run_fe"].rolling(window=8, min_periods=3, center=True).mean()

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.scatter(df["date"], df["run_fe"], s=28, alpha=0.55,
               color="#1f77b4", label="Run fixed effect (ε̂)")
    ax.plot(df["date"], df["fe_roll"], color="#d62728", lw=2,
            label="Rolling mean (8 runs)")

    ax.set_title("Run-level fixed effect over time\n"
                 "HR conditional on pace and elevation — negative ⇒ lower HR than predicted (fitness gain)",
                 fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("ε̂  (bpm, actual − predicted HR)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=140)
    print(f"\nPlot saved to {OUT_PLOT}")

    print("\nRun FE summary (bpm):")
    print(df["run_fe"].describe().round(2))

    print("\nFirst vs last quartile of dates — mean FE:")
    q = pd.qcut(df["date"].astype("int64"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    print(df.groupby(q, observed=True)["run_fe"].mean().round(2))


if __name__ == "__main__":
    main()
