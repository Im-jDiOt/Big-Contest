from __future__ import annotations
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "서울시_상권분석서비스"
OUT = BASE / "data" / "processed"
FIG = BASE / "data" / "fig"
FIG.mkdir(parents=True, exist_ok=True)

PANEL = OUT / "trade_area_features.parquet"
UCSV = OUT / "u_shape_results.csv"
SP_LAG = OUT / "trade_area_spatial_lag.parquet"
SP_STAT = OUT / "spatial_morans_stats.json"
ML_LATEST = OUT / "trade_area_risk_score_ml_latest.parquet"
FACTOR_EN = OUT / "factor_elasticnet_coeff.csv"
FACTOR_GBR = OUT / "factor_gbr_importance.csv"
TA_CSV = RAW / "서울시 상권분석서비스(영역-상권).csv"

sns.set(style="whitegrid", font="Malgun Gothic", rc={"axes.unicode_minus": False})
warnings.filterwarnings("ignore")


def _load_ta_xy(path: Path) -> pd.DataFrame:
    from utils.io import read_csv_smart, find_column
    dfh = read_csv_smart(str(path), nrows=200)
    ta_col = (find_column(dfh, ["상권_코드", "상권코드", "trdar_cd"]) or dfh.columns[2])
    x_col = (find_column(dfh, ["중심좌표_x", "x좌표", "x"]) or dfh.columns[4])
    y_col = (find_column(dfh, ["중심좌표_y", "y좌표", "y"]) or dfh.columns[5])
    use = list({ta_col, x_col, y_col})
    df = read_csv_smart(str(path), usecols=use)
    df = df.rename(columns={ta_col: "trade_area_id", x_col: "ta_x", y_col: "ta_y"})
    for c in ["ta_x", "ta_y"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ta_x", "ta_y"]).drop_duplicates("trade_area_id").reset_index(drop=True)


def latest_snapshot(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    last = panel.groupby("trade_area_id")["period_id"].transform("max")
    snap = panel[last == panel["period_id"]][["trade_area_id"] + cols].copy()
    return snap


def plot_u_shape(panel: pd.DataFrame):
    snap = latest_snapshot(panel, ["closure_rate", "store_density_km2", "hhi", "turnover_rate"]).dropna()

    # helper to plot quad fit scatter
    def _quad(ax, x, y, xlabel, title, turning=True):
        xs = (x - x.mean()) / (x.std() + 1e-9)
        X = np.column_stack([xs, xs**2])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        b, a = coef  # linear, quad
        xs_line = np.linspace(xs.min(), xs.max(), 200)
        yhat = b * xs_line + a * xs_line**2
        # turning point in original scale
        xv_std = -b / (2*a) if a != 0 else np.nan
        xv = xv_std * (x.std() + 1e-9) + x.mean()
        ax.scatter(x, y, s=6, alpha=0.3)
        ax.plot((xs_line * (x.std() + 1e-9) + x.mean()), yhat, color="crimson")
        if turning and np.isfinite(xv):
            ax.axvline(xv, ls="--", color="gray", alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("폐업률")
        ax.set_title(title)

    # density
    fig, ax = plt.subplots(figsize=(8, 5))
    s = snap.dropna(subset=["store_density_km2"])
    _quad(ax, s["store_density_km2"].values, s["closure_rate"].values,
          xlabel="점포 밀도(개/㎢)", title="밀도 vs 폐업률 (U-Shape 검증)")
    fig.tight_layout(); fig.savefig(FIG / "u_shape_density.png", dpi=150)
    plt.close(fig)

    # turnover
    fig, ax = plt.subplots(figsize=(8, 5))
    s = snap.dropna(subset=["turnover_rate"])
    _quad(ax, s["turnover_rate"].values, s["closure_rate"].values,
          xlabel="턴오버율", title="턴오버 vs 폐업률 (2차 회귀)", turning=False)
    fig.tight_layout(); fig.savefig(FIG / "u_shape_turnover.png", dpi=150)
    plt.close(fig)

    # HHI
    fig, ax = plt.subplots(figsize=(8, 5))
    s = snap.dropna(subset=["hhi"])
    _quad(ax, s["hhi"].values, s["closure_rate"].values,
          xlabel="HHI(업종 집중도)", title="HHI vs 폐업률 (2차 회귀)", turning=False)
    fig.tight_layout(); fig.savefig(FIG / "relation_hhi.png", dpi=150)
    plt.close(fig)


def plot_stability():
    path = OUT / "trade_area_closure_stability.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["mean_clo"], df["std_clo"], s=10, alpha=0.5, label="상권")
    ax.scatter(df.loc[df["label_persistent_high"], "mean_clo"], df.loc[df["label_persistent_high"], "std_clo"],
               s=12, color="crimson", label="지속 고폐업률")
    ax.scatter(df.loc[df["label_spiky"], "mean_clo"], df.loc[df["label_spiky"], "std_clo"],
               s=12, color="orange", label="급변 상권")
    ax.set_xlabel("평균 폐업률"); ax.set_ylabel("표준편차")
    ax.set_title("상권별 폐업률 안정성: 평균 vs 변동성")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "closure_stability_scatter.png", dpi=150)
    plt.close(fig)


def plot_spatial_lag(panel: pd.DataFrame):
    if not SP_LAG.exists():
        return
    lag = pd.read_parquet(SP_LAG)
    I = None
    if SP_STAT.exists():
        try:
            I = json.loads(Path(SP_STAT).read_text(encoding="utf-8"))
            I = I.get("moran_I_closure_rate_latest")
        except Exception:
            I = None

    # scatter lag vs value
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.kdeplot(x=lag["closure_rate"], y=lag["closure_rate_spatial_lag"], fill=True, thresh=0.05, levels=30, cmap="Blues", ax=ax)
    ax.set_xlabel("폐업률"); ax.set_ylabel("공간 라그(인접 평균 폐업률)")
    title = "공간 자기상관 (Moran's I={:.3f})".format(I) if I is not None else "공간 자기상관"
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(FIG / "spatial_lag_scatter.png", dpi=150)
    plt.close(fig)

    # map-like scatter by XY if available
    if TA_CSV.exists():
        xy = _load_ta_xy(TA_CSV)
        latest = latest_snapshot(panel, ["closure_rate"]).merge(xy, on="trade_area_id", how="inner").dropna()
        fig, ax = plt.subplots(figsize=(6, 7))
        sc = ax.scatter(latest["ta_x"], latest["ta_y"], c=latest["closure_rate"], s=8, cmap="Reds", alpha=0.8)
        ax.set_title("최신 분기 폐업률(중심좌표 산점)")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, shrink=0.8, label="폐업률")
        fig.tight_layout(); fig.savefig(FIG / "map_scatter_closure_rate.png", dpi=150)
        plt.close(fig)


def plot_risk_vs_closure(panel: pd.DataFrame):
    if not ML_LATEST.exists():
        return
    ml = pd.read_parquet(ML_LATEST)
    latest = latest_snapshot(panel, ["closure_rate", "period_id"]).copy()
    df = ml.merge(latest, on="trade_area_id", how="left")

    # bar by grade
    grp = df.groupby("risk_grade_ml")["closure_rate"].mean().reindex(["A","B","C","D","E"])  # ensure order
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=grp.index, y=grp.values, ax=ax, palette="viridis")
    ax.set_xlabel("위험 등급(ML)"); ax.set_ylabel("평균 폐업률"); ax.set_title("위험 등급별 평균 폐업률")
    fig.tight_layout(); fig.savefig(FIG / "risk_grade_vs_closure.png", dpi=150)
    plt.close(fig)

    # risk score distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["risk_0_100_ml"], bins=30, kde=True, ax=ax, color="#355C7D")
    ax.set_xlabel("리스크 점수(0~100)"); ax.set_title("ML 리스크 점수 분포")
    fig.tight_layout(); fig.savefig(FIG / "risk_score_distribution.png", dpi=150)
    plt.close(fig)


def plot_importance():
    # GBR importance
    if FACTOR_GBR.exists():
        imp = pd.read_csv(FACTOR_GBR)
        imp = imp.dropna().sort_values("importance", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=imp, x="importance", y="feature", ax=ax, palette="mako")
        ax.set_title("GBR 변수 중요도 Top15"); ax.set_xlabel("Permutation Importance")
        fig.tight_layout(); fig.savefig(FIG / "gbr_importance_top15.png", dpi=150)
        plt.close(fig)
    # ElasticNet coefs
    if FACTOR_EN.exists():
        en = pd.read_csv(FACTOR_EN)
        en["abs_coef"] = en["coef"].abs()
        top = en.sort_values("abs_coef", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=top, x="abs_coef", y="feature", ax=ax, palette="rocket")
        ax.set_title("ElasticNet 계수 절대값 Top15"); ax.set_xlabel("|coef|")
        fig.tight_layout(); fig.savefig(FIG / "elasticnet_coef_top15.png", dpi=150)
        plt.close(fig)


def main():
    assert PANEL.exists(), f"panel not found: {PANEL}"
    panel = pd.read_parquet(PANEL)

    plot_u_shape(panel)
    plot_stability()
    plot_spatial_lag(panel)
    plot_risk_vs_closure(panel)
    plot_importance()

    print("Figures saved to:", FIG)


if __name__ == "__main__":
    main()

