from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

PANEL_PATH = OUT / "trade_area_features.parquet"


def build_dataset(panel: pd.DataFrame):
    panel = panel.copy()
    # target: next-period closure rate
    panel = panel.sort_values(["trade_area_id", "period_id"]).reset_index(drop=True)
    panel["closure_rate_next"] = panel.groupby("trade_area_id")["closure_rate"].shift(-1)

    # features: levels + yoy + volatility + structure
    feat_cols = [c for c in panel.columns if any(c.startswith(p) for p in [
        "pop_float", "pop_res", "pop_job", "sales", "store_cnt", "hhi", "store_density_km2", "turnover_rate"
    ]) and (c.endswith("_yoy") or c.endswith("_vol4") or True)]
    # prune target leakage (drop closure_rate itself)
    feat_cols = [c for c in feat_cols if c not in ("closure_rate", "closure_rate_next")]

    # drop rows without target
    data = panel.dropna(subset=["closure_rate_next"]).copy()

    X = data[feat_cols].copy()
    y = data["closure_rate_next"].astype(float).values
    meta = data[["trade_area_id", "period_id", "period_str"]].copy()

    # simple impute
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X, y, meta, feat_cols


def time_split(meta: pd.DataFrame, test_last_n: int = 4):
    # take last N unique periods for test
    periods = sorted(meta["period_id"].dropna().unique().tolist())
    if len(periods) <= test_last_n:
        split_pt = periods[max(0, len(periods) - 2)]  # keep minimal train
    else:
        split_pt = periods[-test_last_n]
    return split_pt


def main():
    assert PANEL_PATH.exists(), f"Panel not found: {PANEL_PATH}. Run trade_area_panel.py first."
    panel = pd.read_parquet(PANEL_PATH)

    X, y, meta, feat_cols = build_dataset(panel)
    split_pt = time_split(meta, test_last_n=4)

    train_idx = meta["period_id"] < split_pt
    test_idx = meta["period_id"] >= split_pt

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = GradientBoostingRegressor(random_state=42, n_estimators=400, max_depth=3, learning_rate=0.05)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    print({"MAE": float(mae), "R2": float(r2), "n_train": int(train_idx.sum()), "n_test": int(test_idx.sum())})

    # importance via permutation on test for robustness
    try:
        imp = permutation_importance(model, X_test, y_test, scoring="neg_mean_absolute_error", n_repeats=5, random_state=42)
        imp_df = pd.DataFrame({"feature": feat_cols, "importance": imp.importances_mean}).sort_values("importance", ascending=False)
    except Exception:
        imp_df = pd.DataFrame({"feature": feat_cols, "importance": model.feature_importances_ if hasattr(model, "feature_importances_") else np.nan})

    # full-period predictions to create score by period
    full_pred = model.predict(X)
    score_raw = pd.Series(full_pred, index=meta.index)

    # scale per-period globally to 0-100 (higher predicted closure = higher risk)
    risk_0_100 = (score_raw - score_raw.min()) / (score_raw.max() - score_raw.min() + 1e-9) * 100

    out = meta.copy()
    out["risk_0_100_ml"] = risk_0_100
    q = out["risk_0_100_ml"].rank(pct=True)
    out["risk_grade_ml"] = pd.cut(q, bins=[0, .2, .4, .6, .8, 1.0], labels=["E","D","C","B","A"], include_lowest=True).astype(str)

    # latest-period snapshot per trade_area
    last_period = out.groupby("trade_area_id")["period_id"].transform("max") == out["period_id"]
    snap = out[last_period][["trade_area_id", "risk_0_100_ml", "risk_grade_ml"]].drop_duplicates("trade_area_id")

    score_path = OUT / "trade_area_risk_score_ml.parquet"
    out_path = OUT / "trade_area_risk_score_ml_latest.parquet"
    out.to_parquet(score_path, index=False)
    snap.to_parquet(out_path, index=False)

    # feature importance export
    imp_path = OUT / "trade_area_risk_feature_importance.csv"
    imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")

    print("saved:", score_path, out_path, imp_path)


if __name__ == "__main__":
    main()

