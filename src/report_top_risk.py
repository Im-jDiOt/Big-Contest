from __future__ import annotations
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "processed"


def main():
    snap_path = OUT / "trade_area_risk_score_snapshot.parquet"
    ml_path = OUT / "trade_area_risk_score_ml_latest.parquet"

    if snap_path.exists():
        snap = pd.read_parquet(snap_path)
        print("[Snapshot] rows:", len(snap))
        print(snap.sort_values("risk_0_100", ascending=False).head(10))
    else:
        print("[Snapshot] file not found:", snap_path)

    if ml_path.exists():
        ml = pd.read_parquet(ml_path)
        print("[ML Latest] rows:", len(ml))
        print(ml.sort_values("risk_0_100_ml", ascending=False).head(10))
    else:
        print("[ML Latest] file not found:", ml_path)


if __name__ == "__main__":
    main()

