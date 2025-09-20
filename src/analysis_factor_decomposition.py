from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "processed"
PANEL = OUT / "trade_area_features.parquet"


def build_xy(panel: pd.DataFrame):
    df = panel.copy()
    df = df.sort_values(["trade_area_id", "period_id"]).reset_index(drop=True)
    df["closure_rate_next"] = df.groupby("trade_area_id")["closure_rate"].shift(-1)

    feat_cols = []
    for v in ["pop_float", "pop_res", "pop_job", "sales"]:
        feat_cols += [v, f"{v}_yoy", f"{v}_vol4"]
    feat_cols += ["store_cnt", "store_density_km2", "turnover_rate", "hhi"]

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = df["closure_rate_next"].astype(float)
    meta = df[["trade_area_id", "period_id", "period_str"]].copy()

    # drop rows with missing target
    keep = y.notna()
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)
    meta = meta.loc[keep].reset_index(drop=True)

    return X, y, meta, feat_cols


def time_split(meta: pd.DataFrame, last_n: int = 4):
    periods = sorted(meta["period_id"].dropna().unique().tolist())
    split = periods[-last_n] if len(periods) > last_n else periods[max(0, len(periods)-2)]
    return split


def run():
    assert PANEL.exists(), f"panel not found: {PANEL}"
    panel = pd.read_parquet(PANEL)
    X, y, meta, feat_cols = build_xy(panel)
    split_pt = time_split(meta, last_n=4)

    train = meta["period_id"] < split_pt
    test = meta["period_id"] >= split_pt

    Xtr, ytr = X.loc[train, :].values, y.loc[train].values
    Xte, yte = X.loc[test, :].values, y.loc[test].values

    # ElasticNet with simple scaling and alpha grid
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    best = None
    best_mae = 1e9
    for a in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:
        for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            mdl = ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000, random_state=42)
            mdl.fit(Xtr_s, ytr)
            pred = mdl.predict(Xte_s)
            mae = mean_absolute_error(yte, pred)
            if mae < best_mae:
                best_mae = mae
                best = (mdl, a, l1)

    en, a, l1 = best
    pred_en = en.predict(Xte_s)
    r2_en = r2_score(yte, pred_en)

    coefs = pd.DataFrame({
        "feature": feat_cols,
        "coef": en.coef_,
    }).sort_values("coef", key=np.abs, ascending=False)
    coefs.to_csv(OUT / "factor_elasticnet_coeff.csv", index=False, encoding="utf-8-sig")

    # GradientBoosting for non-linear importance
    gbr = GradientBoostingRegressor(random_state=42, n_estimators=500, max_depth=3, learning_rate=0.05)
    gbr.fit(Xtr, ytr)
    pred_gbr = gbr.predict(Xte)
    mae_gbr = mean_absolute_error(yte, pred_gbr)
    r2_gbr = r2_score(yte, pred_gbr)

    try:
        imp = permutation_importance(gbr, Xte, yte, n_repeats=5, random_state=42, scoring="neg_mean_absolute_error")
        imp_df = pd.DataFrame({"feature": feat_cols, "importance": imp.importances_mean}).sort_values("importance", ascending=False)
    except Exception:
        imp_df = pd.DataFrame({"feature": feat_cols, "importance": np.nan})
    imp_df.to_csv(OUT / "factor_gbr_importance.csv", index=False, encoding="utf-8-sig")

    summ = {
        "ElasticNet": {"alpha": a, "l1_ratio": l1, "MAE": float(best_mae), "R2": float(r2_en)},
        "GBR": {"MAE": float(mae_gbr), "R2": float(r2_gbr)}
    }
    pd.Series(summ).to_json(OUT / "factor_decomposition_metrics.json")
    print(summ)


if __name__ == "__main__":
    run()
