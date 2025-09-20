from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "서울시_상권분석서비스"
OUT = BASE / "data" / "processed"
PANEL = OUT / "trade_area_features.parquet"
TA = RAW / "서울시 상권분석서비스(영역-상권).csv"

from utils.io import read_csv_smart, find_column


def load_trade_area_xy(path: Path) -> pd.DataFrame:
    head = read_csv_smart(str(path), nrows=200)
    ta_col = find_column(head, ["상권_코드", "상권코드", "trdar_cd", "상권 코드"]) or head.columns[2]
    x_col = find_column(head, ["중심좌표_x", "x좌표", "x"]) or head.columns[4]
    y_col = find_column(head, ["중심좌표_y", "y좌표", "y"]) or head.columns[5]
    use = list({ta_col, x_col, y_col})
    df = read_csv_smart(str(path), usecols=use)
    df = df.rename(columns={ta_col: "trade_area_id", x_col: "ta_x", y_col: "ta_y"})
    for c in ["ta_x", "ta_y"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ta_x", "ta_y"]).drop_duplicates("trade_area_id").reset_index(drop=True)
    return df


def morans_i(x: np.ndarray, W: np.ndarray) -> float:
    x = x.astype(float)
    n = len(x)
    x_bar = x.mean()
    z = x - x_bar
    num = 0.0
    W_sum = W.sum()
    # ensure no self-weights
    np.fill_diagonal(W, 0.0)
    num = (W * (z[:, None] @ z[None, :])).sum()
    den = (z ** 2).sum()
    if den == 0 or W_sum == 0:
        return np.nan
    I = (n / W_sum) * (num / den)
    return float(I)


def run(k_neighbors: int = 8):
    assert PANEL.exists(), f"panel not found: {PANEL}"
    panel = pd.read_parquet(PANEL)
    xy = load_trade_area_xy(TA)

    # latest closure rate per TA
    last_pid = panel.groupby("trade_area_id")["period_id"].transform("max")
    snap = panel[last_pid == panel["period_id"]][["trade_area_id", "closure_rate"]].copy()
    snap = snap.merge(xy, on="trade_area_id", how="inner").dropna(subset=["closure_rate"]).reset_index(drop=True)

    # build KNN weights (symmetrize)
    coords = snap[["ta_x", "ta_y"]].values
    nbr = NearestNeighbors(n_neighbors=min(k_neighbors+1, len(snap)), algorithm="auto")
    nbr.fit(coords)
    dist, idx = nbr.kneighbors(coords)
    n = len(snap)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        # skip self index 0
        for j in idx[i, 1:]:
            W[i, j] = 1.0
            W[j, i] = 1.0

    # spatial lag: average neighbors' closure_rate
    x = snap["closure_rate"].values
    deg = W.sum(axis=1)
    with np.errstate(invalid='ignore'):
        lag = (W @ x) / np.where(deg > 0, deg, np.nan)

    snap["closure_rate_spatial_lag"] = lag

    I = morans_i(x, W.copy())

    # save outputs
    lag_out = OUT / "trade_area_spatial_lag.parquet"
    snap[["trade_area_id", "closure_rate", "closure_rate_spatial_lag"]].to_parquet(lag_out, index=False)

    stats = {"moran_I_closure_rate_latest": I, "n": int(n), "k_neighbors": int(k_neighbors)}
    pd.Series(stats).to_json(OUT / "spatial_morans_stats.json")
    print(stats)


if __name__ == "__main__":
    run()

