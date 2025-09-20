from __future__ import annotations
import pandas as pd
import numpy as np


def make_period(df: pd.DataFrame, year_col: str, quarter_col: str) -> pd.DataFrame:
    df = df.copy()
    y = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    q = pd.to_numeric(df[quarter_col], errors="coerce").astype("Int64")
    df["year"] = y
    df["quarter"] = q
    # integer period like 20201..20204
    df["period_id"] = (df["year"] * 10 + df["quarter"]).astype("Int64")
    df["period_str"] = df.apply(lambda r: f"{int(r['year'])}Q{int(r['quarter'])}" if pd.notna(r['year']) and pd.notna(r['quarter']) else pd.NA, axis=1)
    return df


def yoy_change(group: pd.DataFrame, col: str) -> pd.Series:
    g = group.sort_values("period_id").copy()
    g[f"{col}_yoy"] = g[col] / g[col].shift(4) - 1.0
    return g[f"{col}_yoy"]


def rolling_vol(group: pd.DataFrame, col: str, window: int = 4) -> pd.Series:
    g = group.sort_values("period_id").copy()
    return g[col].rolling(window=window, min_periods=2).std()


def minmax_scale(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(50.0, index=s.index)
    return 100 * (s - mn) / (mx - mn)

