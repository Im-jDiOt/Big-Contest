from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Optional

from utils.io import read_csv_smart, find_column
from utils.features import make_period, yoy_change, rolling_vol, minmax_scale

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "서울시_상권분석서비스"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------

def _detect_period_cols(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    y = find_column(df, ["기준_년", "기준_연도", "연도", "년도", "년", "year", "YYYY", "BASE_YR"])
    q = find_column(df, ["기준_분기", "분기", "quarter", "QQ", "BASE_QT", "분기코드"])
    return y, q


def _detect_ta_col(df: pd.DataFrame) -> str:
    return find_column(df, ["상권_코드", "상권코드", "trdar_cd", "상권 코드", "코드", "상권ID", "ID"]) or df.columns[2]


def _numeric_sum(df: pd.DataFrame, include_sub: List[str]) -> pd.Series:
    cols = [c for c in df.columns if any(s in c for s in include_sub)]
    if not cols:
        return pd.Series(0, index=df.index, dtype=float)
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    return num.sum(axis=1)


def _standardize_period(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'year' and 'quarter' columns by detecting Y/Q or YM/M columns flexibly.
    Priority: 연월 -> (연,월) -> (연,분기). If only 월 exists with no 연, fails.
    """
    work = df.copy()
    y_col, q_col = _detect_period_cols(work)
    ym_col = find_column(work, ["기준_연월", "연월", "기준연월", "base_ym", "YYYYMM", "년월", "연_월"])
    m_col = find_column(work, ["기준_월", "월", "month", "MM"]) if q_col is None else None

    if ym_col is not None:
        raw = work[ym_col].astype(str)
        # try to extract 6 digits YYYYMM
        s = raw.str.extract(r"(?P<ym>(?:19|20)\d{2}[\-/]?(?:0[1-9]|1[0-2]))", expand=False)[0]
        # fallback: remove non-digits
        s = s.fillna(raw.str.replace(r"\D", "", regex=True))
        y = pd.to_numeric(s.str.slice(0, 4), errors="coerce").astype("Int64")
        m = pd.to_numeric(s.str.slice(-2), errors="coerce").astype("Int64")
        q = ((m - 1) // 3 + 1).astype("Int64")
        work["year"], work["quarter"] = y, q
    elif y_col is not None and q_col is not None:
        y = pd.to_numeric(work[y_col], errors="coerce").astype("Int64")
        q = pd.to_numeric(work[q_col], errors="coerce").astype("Int64")
        work["year"], work["quarter"] = y, q
    elif y_col is not None and m_col is not None:
        y = pd.to_numeric(work[y_col], errors="coerce").astype("Int64")
        m = pd.to_numeric(work[m_col], errors="coerce").astype("Int64")
        q = ((m - 1) // 3 + 1).astype("Int64")
        work["year"], work["quarter"] = y, q
    else:
        raise ValueError(f"기간 컬럼 탐지 실패: year={y_col}, quarter={q_col}, ym={ym_col}, month={m_col}. cols={list(work.columns)[:30]}")

    return work

# ---------- Loaders ----------

def load_trade_area_master(path: Path) -> pd.DataFrame:
    df = read_csv_smart(str(path), nrows=200)
    ta_col = _detect_ta_col(df)
    x_col = find_column(df, ["중심좌표_x", "x좌표", "x", "중심x", "좌표x"]) or df.columns[4]
    y_col = find_column(df, ["중심좌표_y", "y좌표", "y", "중심y", "좌표y"]) or df.columns[5]
    area_col = find_column(df, ["면적", "상권_면적", "영역면적", "area", "면적(m2)"])
    use = list({ta_col, x_col, y_col} | ({area_col} if area_col else set()))
    full = read_csv_smart(str(path), usecols=use)
    full = full.rename(columns={ta_col: "trade_area_id", x_col: "ta_x", y_col: "ta_y"})
    if area_col: full = full.rename(columns={area_col: "area_m2"})
    for c in ["ta_x", "ta_y", "area_m2"]:
        if c in full.columns: full[c] = pd.to_numeric(full[c], errors="coerce")
    full = full.dropna(subset=["ta_x", "ta_y"]).drop_duplicates("trade_area_id").reset_index(drop=True)
    return full


def load_population(path: Path, key_substr: List[str]) -> pd.DataFrame:
    df = read_csv_smart(str(path))
    ta_col = _detect_ta_col(df)
    dfp = _standardize_period(df)
    val = _numeric_sum(df, key_substr)
    out = pd.DataFrame({
        "trade_area_id": df[ta_col].values,
        "year": dfp["year"].values,
        "quarter": dfp["quarter"].values,
        "value": val.values,
    })
    out = make_period(out, "year", "quarter")
    out = out.groupby(["trade_area_id", "year", "quarter", "period_id", "period_str"], as_index=False)["value"].sum()
    return out


def load_sales(files: List[Path]) -> pd.DataFrame:
    frames = []
    for p in files:
        df = read_csv_smart(str(p))
        ta_col = _detect_ta_col(df)
        dfp = _standardize_period(df)
        cand_cols = [c for c in df.columns if any(s in c for s in ["매출", "금액", "추정"]) and df[c].dtype != 'O']
        if not cand_cols:
            cand_cols = [c for c in df.columns if any(s in c for s in ["매출", "금액", "추정"])]
        val = df[cand_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1) if cand_cols else pd.Series(0, index=df.index, dtype=float)
        out = pd.DataFrame({
            "trade_area_id": df[ta_col].values,
            "year": dfp["year"].values,
            "quarter": dfp["quarter"].values,
            "sales": val.values,
        })
        out = make_period(out, "year", "quarter")
        frames.append(out[["trade_area_id", "year", "quarter", "period_id", "period_str", "sales"]])
    if not frames:
        return pd.DataFrame(columns=["trade_area_id", "year", "quarter", "period_id", "period_str", "sales"])
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.groupby(["trade_area_id", "year", "quarter", "period_id", "period_str"], as_index=False)["sales"].sum()
    return all_df


def load_store(file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_csv_smart(str(file))
    ta_col = _detect_ta_col(df)
    dfp = _standardize_period(df)
    # store count columns
    candidates = [c for c in df.columns if ("점포" in c and ("수" in c or "합계" in c)) or ("사업체" in c and "수" in c)]
    val = df[candidates].apply(pd.to_numeric, errors="coerce").sum(axis=1) if candidates else pd.Series(1, index=df.index)
    out = pd.DataFrame({
        "trade_area_id": df[ta_col].values,
        "year": dfp["year"].values,
        "quarter": dfp["quarter"].values,
        "store_cnt": pd.to_numeric(val, errors="coerce").values,
    })
    out = make_period(out, "year", "quarter")
    out = out.groupby(["trade_area_id", "year", "quarter", "period_id", "period_str"], as_index=False)["store_cnt"].sum()

    # category for HHI if exists
    cat_col = find_column(df, ["업종", "업종명", "서비스업종", "중분류", "소분류", "category"])  # optional
    if cat_col:
        dfc = pd.DataFrame({
            "trade_area_id": df[ta_col].values,
            "period_id": (dfp["year"] * 10 + dfp["quarter"]).astype("Int64").values,
            "category": df[cat_col].astype(str).values,
        })
        grp = dfc.groupby(["trade_area_id", "period_id", "category"]).size().rename("cnt").reset_index()
        tot = grp.groupby(["trade_area_id", "period_id"]).agg(tot=("cnt", "sum")).reset_index()
        grp = grp.merge(tot, on=["trade_area_id", "period_id"], how="left")
        grp["share"] = grp["cnt"] / grp["tot"].replace(0, np.nan)
        hhi = grp.assign(share_sq=lambda d: d["share"] ** 2).groupby(["trade_area_id", "period_id"]).agg(hhi=("share_sq", "sum")).reset_index()
    else:
        hhi = pd.DataFrame(columns=["trade_area_id", "period_id", "hhi"])

    return out, hhi

# ---------- Build panel ----------

def build_panel() -> pd.DataFrame:
    ta_master = load_trade_area_master(RAW / "서울시 상권분석서비스(영역-상권).csv")

    pop_float = load_population(RAW / "서울시 상권분석서비스(길단위인구-상권).csv", ["유동", "이동", "유입"])  # proxy
    pop_res = load_population(RAW / "서울시 상권분석서비스(상주인구-상권).csv", ["상주", "주거", "거주", "인구"])
    pop_job = load_population(RAW / "서울시 상권분석서비스(직장인구-상권).csv", ["직장", "종사", "근로", "인구"])

    sales_files = [
        RAW / "서울시_상권분석서비스(추정매출-상권)_2020년.csv",
        RAW / "서울시_상권분석서비스(추정매출-상권)_2021년.csv",
        RAW / "서울시_상권분석서비스(추정매출-상권)_2022년.csv",
        RAW / "서울시_상권분석서비스(추정매출-상권)_2023년.csv",
        RAW / "서울시 상권분석서비스(추정매출-상권)_2024년.csv",
    ]
    sales_files = [p for p in sales_files if p.exists()]
    sales = load_sales(sales_files)

    store, hhi = load_store(RAW / "서울시 상권분석서비스(점포-상권).csv")

    # base periods
    periods = pd.concat([
        pop_float[["trade_area_id", "period_id", "period_str"]],
        pop_res[["trade_area_id", "period_id", "period_str"]],
        pop_job[["trade_area_id", "period_id", "period_str"]],
        sales[["trade_area_id", "period_id", "period_str"]],
        store[["trade_area_id", "period_id", "period_str"]],
    ], ignore_index=True).drop_duplicates()

    panel = periods.copy()
    panel = panel.merge(pop_float.rename(columns={"value": "pop_float"}), on=["trade_area_id", "period_id", "period_str"], how="left")
    panel = panel.merge(pop_res.rename(columns={"value": "pop_res"}), on=["trade_area_id", "period_id", "period_str"], how="left")
    panel = panel.merge(pop_job.rename(columns={"value": "pop_job"}), on=["trade_area_id", "period_id", "period_str"], how="left")
    panel = panel.merge(sales[["trade_area_id", "period_id", "period_str", "sales"]], on=["trade_area_id", "period_id", "period_str"], how="left")
    panel = panel.merge(store[["trade_area_id", "period_id", "period_str", "store_cnt"]], on=["trade_area_id", "period_id", "period_str"], how="left")
    panel = panel.merge(hhi, on=["trade_area_id", "period_id"], how="left")

    # attach static area for density
    ta_area = load_trade_area_master(RAW / "서울시 상권분석서비스(영역-상권).csv")[["trade_area_id", "area_m2"]]
    panel = panel.merge(ta_area, on="trade_area_id", how="left")
    panel["store_density_km2"] = panel.apply(lambda r: (r["store_cnt"] / (r["area_m2"] / 1e6)) if pd.notna(r.get("area_m2")) and pd.notna(r.get("store_cnt")) and r.get("area_m2") not in (0, None) else np.nan, axis=1)

    # sort and lag features
    panel = panel.sort_values(["trade_area_id", "period_id"]).reset_index(drop=True)
    # lag store count to compute closure/opening proxies
    panel["store_cnt_prev"] = panel.groupby("trade_area_id")["store_cnt"].shift(1)
    diff = panel["store_cnt"] - panel["store_cnt_prev"]
    panel["openings"] = diff.clip(lower=0).fillna(0)
    panel["closures"] = (-diff).clip(lower=0).fillna(0)
    panel["closure_rate"] = np.where(panel["store_cnt_prev"].fillna(0) > 0, panel["closures"] / panel["store_cnt_prev"], np.nan)
    panel["turnover_rate"] = np.where(((panel["store_cnt"] + panel["store_cnt_prev"]).fillna(0) / 2) > 0,
                                       (panel["openings"] + panel["closures"]) / ((panel["store_cnt"] + panel["store_cnt_prev"]) / 2), np.nan)

    # YoY and volatility for demand/sales
    for v in ["pop_float", "pop_res", "pop_job", "sales"]:
        panel[f"{v}_yoy"] = panel.groupby("trade_area_id").apply(lambda g: yoy_change(g, v)).reset_index(level=0, drop=True)
        panel[f"{v}_vol4"] = panel.groupby("trade_area_id").apply(lambda g: rolling_vol(g, v, 4)).reset_index(level=0, drop=True)

    # export panels
    panel_out = OUT / "trade_area_features.parquet"
    panel.to_parquet(panel_out, index=False)

    # export closure rates for analysis
    clo = panel[["trade_area_id", "period_id", "period_str", "closure_rate", "turnover_rate"]].copy()
    clo_out = OUT / "trade_area_closure_rate.parquet"
    clo.to_parquet(clo_out, index=False)

    return panel


if __name__ == "__main__":
    build_panel()
