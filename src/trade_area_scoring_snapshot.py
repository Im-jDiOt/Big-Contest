from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from utils.io import read_csv_smart, find_column

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

STORE_PATH = BASE / "data1_hjd_coord.csv"
TA_PATH = RAW / "서울시_상권분석서비스" / "서울시 상권분석서비스(영역-상권).csv"


def load_trade_areas(path: Path) -> pd.DataFrame:
    # Read a small sample to detect columns
    head = read_csv_smart(str(path), nrows=200)
    # infer columns
    ta_id_col = find_column(head, [
        "상권_코드", "상권코드", "trdar_cd", "상권 코드", "코드", "상권_�드", "상권ID", "ID"
    ]) or head.columns[2]
    x_col = find_column(head, ["중심좌표_x", "x좌표", "x", "중심x", "좌표x"]) or head.columns[4]
    y_col = find_column(head, ["중심좌표_y", "y좌표", "y", "중심y", "좌표y"]) or head.columns[5]
    area_col = find_column(head, ["면적", "상권_면적", "영역면적", "area", "면적(m2)"])

    usecols = list({ta_id_col, x_col, y_col} | ({area_col} if area_col else set()))
    df = read_csv_smart(str(path), usecols=usecols)
    df = df.rename(columns={ta_id_col: "trade_area_id", x_col: "ta_x", y_col: "ta_y"})
    if area_col:
        df = df.rename(columns={area_col: "area_m2"})
    # clean types
    for c in ["ta_x", "ta_y", "area_m2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # drop rows missing coords
    df = df.dropna(subset=["ta_x", "ta_y"]).copy()
    # de-duplicate trade_area_id by first occurrence
    df = df.drop_duplicates(subset=["trade_area_id"]).reset_index(drop=True)
    return df


def load_stores(path: Path) -> pd.DataFrame:
    head = read_csv_smart(str(path), nrows=200)
    x_col = find_column(head, ["x", "X", "경도", "x좌표", "tm_x", "POINT_X", "X좌표"])
    y_col = find_column(head, ["y", "Y", "위도", "y좌표", "tm_y", "POINT_Y", "Y좌표"])
    id_col = find_column(head, ["store_id", "id", "가맹점id", "점포id", "일련번호"]) or head.columns[0]
    cat_col = find_column(head, ["업종", "업종명", "업종대분류", "업종소분류", "category", "HPSN_MCT_BZN_CD_NM"])  # optional

    candidates = {id_col}
    if x_col: candidates.add(x_col)
    if y_col: candidates.add(y_col)
    if cat_col: candidates.add(cat_col)

    df = read_csv_smart(str(path), usecols=list(candidates))
    rename_map = {id_col: "store_id"}
    if x_col: rename_map[x_col] = "x"
    if y_col: rename_map[y_col] = "y"
    if cat_col: rename_map[cat_col] = "category"
    df = df.rename(columns=rename_map)
    # numeric coords
    df["x"] = pd.to_numeric(df.get("x"), errors="coerce")
    df["y"] = pd.to_numeric(df.get("y"), errors="coerce")
    # drop missing coords
    df = df.dropna(subset=["x", "y"]).copy()
    # fill category
    if "category" not in df.columns:
        df["category"] = "ALL"
    # clean IDs
    df["store_id"] = df["store_id"].astype(str)
    return df


def nearest_map(stores: pd.DataFrame, areas: pd.DataFrame) -> pd.DataFrame:
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nbrs.fit(areas[["ta_x", "ta_y"]].values)
    dist, idx = nbrs.kneighbors(stores[["x", "y"]].values)
    mapped = stores.copy()
    mapped["trade_area_id"] = areas.iloc[idx.flatten()]["trade_area_id"].values
    mapped["nn_dist"] = dist.flatten()
    return mapped


def make_competition_features(mapped: pd.DataFrame, areas: pd.DataFrame) -> pd.DataFrame:
    # counts per TA
    ta_store = mapped.groupby("trade_area_id").size().rename("store_cnt_total").reset_index()
    # HHI by category
    grp = mapped.groupby(["trade_area_id", "category"]).size().rename("cnt").reset_index()
    tot = grp.groupby("trade_area_id")["cnt"].sum().rename("tot").reset_index()
    grp = grp.merge(tot, on="trade_area_id", how="left")
    grp["share"] = grp["cnt"] / grp["tot"].replace(0, np.nan)
    hhi = grp.assign(share_sq=grp["share"] ** 2).groupby("trade_area_id")["share_sq"].sum().rename("hhi").reset_index()

    feat = areas[["trade_area_id"]].merge(ta_store, on="trade_area_id", how="left").merge(hhi, on="trade_area_id", how="left")
    feat["store_cnt_total"] = feat["store_cnt_total"].fillna(0).astype(int)
    feat["hhi"] = feat["hhi"].fillna(0.0)

    # density if area available
    if "area_m2" in areas.columns:
        feat = feat.merge(areas[["trade_area_id", "area_m2"]], on="trade_area_id", how="left")
        feat["store_density_km2"] = feat.apply(lambda r: (r["store_cnt_total"] / (r["area_m2"] / 1e6)) if pd.notna(r.get("area_m2")) and r.get("area_m2") not in (0, None) else np.nan, axis=1)
    else:
        feat["store_density_km2"] = np.nan

    return feat


def score_and_grade(feat: pd.DataFrame) -> pd.DataFrame:
    work = feat.copy()
    # standardize selected variables
    score_vars = []
    if "hhi" in work.columns: score_vars.append("hhi")
    if "store_density_km2" in work.columns: score_vars.append("store_density_km2")
    if not score_vars:
        work["risk_0_100"] = 50.0
        work["risk_grade"] = "C"
        return work

    scaler = RobustScaler()
    std_cols = [f"{c}_std" for c in score_vars]
    work[std_cols] = scaler.fit_transform(work[score_vars])
    # weights
    weight_map = {"hhi_std": 0.6, "store_density_km2_std": 0.4}
    work["risk_raw"] = 0.0
    for c in std_cols:
        w = weight_map.get(c, 0.0)
        work["risk_raw"] += work[c] * w
    # 0-100
    rmin, rmax = work["risk_raw"].min(), work["risk_raw"].max()
    if pd.isna(rmin) or pd.isna(rmax) or rmax == rmin:
        work["risk_0_100"] = 50.0
    else:
        work["risk_0_100"] = (work["risk_raw"] - rmin) / (rmax - rmin) * 100
    # grades (quintiles)
    q = work["risk_0_100"].rank(pct=True)
    work["risk_grade"] = pd.cut(q, bins=[0, .2, .4, .6, .8, 1.0], labels=["E","D","C","B","A"], include_lowest=True).astype(str)
    return work


def main():
    assert STORE_PATH.exists(), f"Store file not found: {STORE_PATH}"
    assert TA_PATH.exists(), f"Trade area file not found: {TA_PATH}"

    print("[1/4] Load trade areas…")
    ta = load_trade_areas(TA_PATH)
    print(f"  trade areas: {len(ta):,}")

    print("[2/4] Load stores…")
    stores = load_stores(STORE_PATH)
    print(f"  stores with coords: {len(stores):,}")

    print("[3/4] Nearest mapping store→trade_area…")
    mapped = nearest_map(stores, ta)
    # save mapping
    map_out = OUT / "store_to_trade_area.parquet"
    mapped.drop(columns=[], errors="ignore").to_parquet(map_out, index=False)
    print(f"  saved: {map_out}")

    print("[4/4] Features & scoring…")
    feat = make_competition_features(mapped, ta)
    scored = score_and_grade(feat)
    score_out = OUT / "trade_area_risk_score_snapshot.parquet"
    keep_cols = [c for c in ["trade_area_id", "store_cnt_total", "hhi", "store_density_km2", "risk_0_100", "risk_grade"] if c in scored.columns]
    scored[keep_cols].to_parquet(score_out, index=False)
    print(f"  saved: {score_out}")

    print("DONE")


if __name__ == "__main__":
    main()

