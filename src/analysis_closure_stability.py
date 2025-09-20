from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "processed"
PANEL = OUT / "trade_area_features.parquet"


def _trend_slope(y: pd.Series) -> float:
    y = y.dropna()
    if len(y) < 3:
        return np.nan
    x = np.arange(len(y))
    # center to reduce numeric issues
    x = x - x.mean()
    yv = y.values
    denom = (x**2).sum()
    if denom == 0:
        return 0.0
    slope = (x * yv).sum() / denom
    return float(slope)


def main():
    assert PANEL.exists(), f"panel not found: {PANEL}"
    df = pd.read_parquet(PANEL)
    # focus on closure_rate
    use = df[["trade_area_id", "period_id", "closure_rate"]].copy()
    # aggregate per trade area
    agg = (use.groupby("trade_area_id")
              .agg(
                  n_periods=("closure_rate", lambda s: int(s.notna().sum())),
                  mean_clo=("closure_rate", "mean"),
                  std_clo=("closure_rate", "std"),
                  iqr_clo=("closure_rate", lambda s: (s.quantile(0.75) - s.quantile(0.25))),
              )
              .reset_index())
    # coefficient of variation
    agg["cv_clo"] = agg["std_clo"] / agg["mean_clo"].replace(0, np.nan)

    # trend slope per TA
    slopes = (use.sort_values(["trade_area_id", "period_id"])
                 .groupby("trade_area_id")["closure_rate"]
                 .apply(_trend_slope)
                 .rename("slope_clo")
                 .reset_index())
    agg = agg.merge(slopes, on="trade_area_id", how="left")

    # shock index: count of period-to-period change above 1.5*IQR of TA series
    def _shock_count(s: pd.Series) -> int:
        s = s.dropna()
        if len(s) < 4:
            return 0
        dif = s.diff()
        iqr = (s.quantile(0.75) - s.quantile(0.25))
        thr = 1.5 * (iqr if iqr and iqr > 0 else (s.std() or 0))
        if thr == 0 or np.isnan(thr):
            thr = s.std() * 1.5 if s.std() > 0 else 0.05
        return int((dif.abs() > thr).sum())

    shocks = (use.sort_values(["trade_area_id", "period_id"])
                 .groupby("trade_area_id")["closure_rate"]
                 .apply(_shock_count)
                 .rename("shock_cnt")
                 .reset_index())
    agg = agg.merge(shocks, on="trade_area_id", how="left")

    # labels
    q_mean75 = agg["mean_clo"].quantile(0.75)
    q_std75 = agg["std_clo"].quantile(0.75)
    q_slope75 = agg["slope_clo"].quantile(0.75)

    agg["label_persistent_high"] = (agg["mean_clo"] >= q_mean75) & (agg["std_clo"].fillna(0) <= q_std75)
    agg["label_spiky"] = agg["shock_cnt"] >= agg["shock_cnt"].quantile(0.75)
    agg["label_trending_up"] = agg["slope_clo"].fillna(0) >= q_slope75

    out_path = OUT / "trade_area_closure_stability.parquet"
    agg.to_parquet(out_path, index=False)

    # quick tops CSV
    tops = agg.sort_values(["mean_clo", "std_clo"], ascending=[False, True]).head(50)
    tops.to_csv(OUT / "closure_stability_top.csv", index=False, encoding="utf-8-sig")

    print({
        "rows": int(len(agg)),
        "persistent_high": int(agg["label_persistent_high"].sum()),
        "spiky": int(agg["label_spiky"].sum()),
        "trending_up": int(agg["label_trending_up"].sum()),
    })


if __name__ == "__main__":
    main()

