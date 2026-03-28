"""
OLS regression: average_heartrate ~ pace + elevation + period dummies

Tests whether heart rate has been decreasing over time, controlling for
effort (pace) and terrain (elevation). A negative coefficient on later
period dummies means improved cardiovascular fitness.
"""

import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("runs_panel.csv", parse_dates=["date"])

    # Drop rows without heart rate data
    df = df.dropna(subset=["average_heartrate", "pace_min_per_km", "total_elevation_gain_m"])
    print(f"Observations with HR data: {len(df)}")

    if len(df) < 10:
        print("Not enough observations for meaningful regression.")
        return

    # Create period dummies (quarterly)
    df["period"] = df["date"].dt.to_period("Q").astype(str)
    periods_sorted = sorted(df["period"].unique())
    print(f"\nPeriods: {periods_sorted}")
    print(f"Base period (omitted): {periods_sorted[0]}\n")

    # Dummies — drop first period as reference
    period_dummies = pd.get_dummies(df["period"], prefix="Q", drop_first=True, dtype=float)

    # Build X matrix
    X = pd.concat([
        df[["pace_min_per_km", "total_elevation_gain_m"]].reset_index(drop=True),
        period_dummies.reset_index(drop=True),
    ], axis=1)
    X = sm.add_constant(X)

    y = df["average_heartrate"].reset_index(drop=True)

    # OLS
    model = sm.OLS(y, X).fit(cov_type="HC1")  # robust standard errors
    print(model.summary())

    # Interpretation helper
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    period_cols = [c for c in model.params.index if c.startswith("Q_")]
    if period_cols:
        last_period = period_cols[-1]
        coef = model.params[last_period]
        pval = model.pvalues[last_period]
        direction = "lower" if coef < 0 else "higher"
        sig = "significant" if pval < 0.05 else "not significant"
        print(f"\nLatest period ({last_period}) vs base:")
        print(f"  HR is {abs(coef):.1f} bpm {direction} (p={pval:.3f}, {sig})")
        print(f"  Controlling for pace and elevation gain.")

    print(f"\nPace effect: {model.params['pace_min_per_km']:.2f} bpm per min/km")
    print(f"Elevation effect: {model.params['total_elevation_gain_m']:.3f} bpm per meter of gain")


if __name__ == "__main__":
    main()
