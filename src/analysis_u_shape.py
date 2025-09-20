from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "processed"
PANEL = OUT / "trade_area_features.parquet"


def quad_fit(x: np.ndarray, y: np.ndarray):
    X = np.column_stack([x, x**2])
    mdl = LinearRegression()
    mdl.fit(X, y)
    yhat = mdl.predict(X)
    r2 = r2_score(y, yhat)
    a2 = mdl.coef_[1]
    # U-shape if quadratic term > 0 and significant magnitude vs linear term
    u_shape = a2 > 0
    return mdl, r2, u_shape


def run():
    assert PANEL.exists(), f"panel not found: {PANEL}"
    df = pd.read_parquet(PANEL)
    # latest period per TA
    latest_pid = df.groupby("trade_area_id")["period_id"].transform("max")
    snap = df[df["period_id"] == latest_pid]

    # variables to test
    tests = {
        "store_density_km2": "밀도",
        "hhi": "HHI",
        "turnover_rate": "턴오버",
    }
    rows = []
    for var, label in tests.items():
        s = snap[[var, "closure_rate"]].dropna()
        if len(s) < 50:
            continue
        x = s[var].values.astype(float)
        y = s["closure_rate"].values.astype(float)
        # standardize x for numerical stability
        xs = (x - x.mean()) / (x.std() + 1e-9)
        mdl, r2, uflag = quad_fit(xs, y)
        # compute vertex (turning point) in original scale: x* = -b/(2a)
        a = mdl.coef_[1]
        b = mdl.coef_[0]
        xv_std = -b / (2*a) if a != 0 else np.nan
        xv = xv_std * (x.std() + 1e-9) + x.mean()
        rows.append({
            "feature": var,
            "label": label,
            "n": len(s),
            "R2_quad": float(r2),
            "is_u_shape": bool(uflag),
            "turning_point": float(xv) if np.isfinite(xv) else np.nan,
            "coef_lin": float(b),
            "coef_quad": float(a),
        })

    out = pd.DataFrame(rows)
    out_path = OUT / "u_shape_results.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print({"rows": int(len(out)), "path": str(out_path)})


if __name__ == "__main__":
    run()

