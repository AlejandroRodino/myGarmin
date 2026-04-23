"""
Fitness metric: conditional HR trend over time.

Model: HR_i = α + β·pace_i + γ·elev_i + Σ_t δ_t·bin_t + ε_i

The time fixed effects δ_t capture the average HR in time bin t net of
pace and elevation. A declining δ_t = lower HR at the same effort = fitter.

Usable from the CLI (prints summary + saves plot) or imported as a module
(`run_analysis()` returns a dict of results and also writes the PNG).
"""

import os
from datetime import timedelta

import matplotlib

matplotlib.use("Agg")  # headless backend for server deployment

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = os.path.dirname(__file__)
PANEL = os.path.join(BASE_DIR, "runs_panel.csv")
OUT_PLOT = os.path.join(BASE_DIR, "fitness_trend.png")

# Analysis window and time-bin granularity
START_DATE = pd.Timestamp("2026-03-01")   # inclusive; set to None for no cutoff
BIN_DAYS = 14                              # biweekly

# Outlier thresholds
MIN_DISTANCE_KM = 0.5
PACE_BOUNDS = (3.0, 10.0)  # min/km


def run_analysis(panel_path: str = PANEL, out_plot: str = OUT_PLOT) -> dict:
    """Fit the HR regression with biweekly FEs. Return a results dict and save the plot."""
    df = pd.read_csv(panel_path, parse_dates=["date"])
    df = df[df["activity_type"] == "Run"].copy()
    df = df.dropna(subset=["average_heartrate", "pace_min_per_km", "total_elevation_gain_m"])
    df = df[df["distance_km"] >= MIN_DISTANCE_KM]
    df = df[df["pace_min_per_km"].between(*PACE_BOUNDS)]
    if START_DATE is not None:
        df = df[df["date"] >= START_DATE]
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 4:
        raise ValueError(
            f"Not enough runs after filtering (n={len(df)}). "
            f"Need at least 4 for biweekly FEs + 2 controls."
        )

    # Anchor bins at START_DATE (or min date) so bin boundaries are stable across runs
    anchor = (START_DATE or df["date"].min()).normalize()
    df["bin_idx"] = ((df["date"] - anchor).dt.days // BIN_DAYS).astype(int)
    df["bin_start"] = df["bin_idx"].apply(lambda k: anchor + timedelta(days=int(k) * BIN_DAYS))
    df["bin_label"] = df["bin_start"].dt.strftime("%Y-%m-%d")

    bins_sorted = sorted(df["bin_label"].unique())
    base_bin = bins_sorted[0]
    bin_counts = df.groupby("bin_label").size().to_dict()

    bin_dummies = pd.get_dummies(df["bin_label"], prefix="b", drop_first=False, dtype=float)
    bin_dummies = bin_dummies.drop(columns=[f"b_{base_bin}"])

    X = pd.concat([
        df[["pace_min_per_km", "total_elevation_gain_m"]].reset_index(drop=True),
        bin_dummies.reset_index(drop=True),
    ], axis=1)
    X = sm.add_constant(X)
    y = df["average_heartrate"].reset_index(drop=True)

    model = sm.OLS(y, X).fit(cov_type="HC1")

    fe_rows = []
    for b in bins_sorted:
        col = f"b_{b}"
        if b == base_bin:
            fe_rows.append({
                "bin": b, "coef": 0.0, "lo": 0.0, "hi": 0.0,
                "pvalue": None, "n_runs": int(bin_counts.get(b, 0)),
            })
        else:
            coef = float(model.params[col])
            se = float(model.bse[col])
            fe_rows.append({
                "bin": b,
                "coef": coef,
                "lo": coef - 1.96 * se,
                "hi": coef + 1.96 * se,
                "pvalue": float(model.pvalues[col]),
                "n_runs": int(bin_counts.get(b, 0)),
            })
    fe_df = pd.DataFrame(fe_rows)
    fe_df["date"] = pd.to_datetime(fe_df["bin"]) + pd.Timedelta(days=BIN_DAYS // 2)
    fe_df["t"] = np.arange(len(fe_df))

    # Linear trend on the FE sequence
    X_trend = sm.add_constant(fe_df["t"])
    trend = sm.OLS(fe_df["coef"], X_trend).fit(cov_type="HC1")
    slope = float(trend.params["t"])
    slope_p = float(trend.pvalues["t"])
    per_month = slope * (30 / BIN_DAYS)

    if slope < 0 and slope_p < 0.10:
        verdict = "Evidence of fitness improvement (HR falling at constant effort)."
    elif slope > 0 and slope_p < 0.10:
        verdict = "HR trending UP at constant effort — not getting fitter."
    else:
        verdict = "No statistically meaningful trend (flat within noise)."

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhline(0, color="#888", lw=0.8, ls="--", label=f"Reference ({base_bin})")
    ax.errorbar(
        fe_df["date"], fe_df["coef"],
        yerr=[fe_df["coef"] - fe_df["lo"], fe_df["hi"] - fe_df["coef"]],
        fmt="o", color="#1f77b4", ecolor="#1f77b4", alpha=0.8,
        capsize=3, markersize=7, label="Biweekly FE (95% CI)",
    )
    fe_df["fitted_trend"] = trend.fittedvalues
    ax.plot(fe_df["date"], fe_df["fitted_trend"], color="#d62728", lw=2,
            label=f"Linear trend: {per_month:+.2f} bpm/month (p={slope_p:.2f})")
    ax.set_title(f"Conditional HR by biweek (from {df['date'].min().date()})\n"
                 "Lower = lower HR at the same pace & elevation → fitter",
                 fontsize=11)
    ax.set_xlabel("Biweek midpoint")
    ax.set_ylabel("Biweek fixed effect (bpm, vs reference)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=BIN_DAYS))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_plot, dpi=140)
    plt.close(fig)

    return {
        "n_runs": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "n_bins": int(len(bins_sorted)),
        "base_bin": base_bin,
        "bins": fe_rows,
        "regression": {
            "r_squared": float(model.rsquared),
            "const": float(model.params["const"]),
            "pace_coef": float(model.params["pace_min_per_km"]),
            "pace_p": float(model.pvalues["pace_min_per_km"]),
            "elev_coef": float(model.params["total_elevation_gain_m"]),
            "elev_p": float(model.pvalues["total_elevation_gain_m"]),
        },
        "trend": {
            "slope_per_bin": slope,
            "slope_per_month": per_month,
            "pvalue": slope_p,
        },
        "verdict": verdict,
        "plot_path": out_plot,
    }


def main():
    res = run_analysis()
    print(f"n = {res['n_runs']} runs  {res['date_min']} → {res['date_max']}")
    print(f"{res['n_bins']} biweekly bins; reference = {res['base_bin']}\n")
    print(f"Pace coef: {res['regression']['pace_coef']:+.2f} (p={res['regression']['pace_p']:.3f})")
    print(f"Elev coef: {res['regression']['elev_coef']:+.3f} (p={res['regression']['elev_p']:.3f})")
    print(f"R²: {res['regression']['r_squared']:.3f}\n")
    print("Biweekly FEs (bpm vs ref):")
    for b in res["bins"]:
        p = f"p={b['pvalue']:.3f}" if b["pvalue"] is not None else "ref"
        print(f"  {b['bin']}  coef={b['coef']:+7.2f}  [{b['lo']:+7.2f}, {b['hi']:+7.2f}]  "
              f"n={b['n_runs']}  {p}")
    print(f"\nTrend: {res['trend']['slope_per_month']:+.2f} bpm/month "
          f"(p={res['trend']['pvalue']:.3f})")
    print(f"⇒ {res['verdict']}")
    print(f"\nPlot saved to {res['plot_path']}")


if __name__ == "__main__":
    main()
