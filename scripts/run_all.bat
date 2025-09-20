@echo off
setlocal
cd /d %~dp0\..

REM 1) 의존성 설치
python -m pip install -r requirements.txt || goto :eof

REM 2) 스냅샷 스코어(최근접 상권 매핑→HHI/밀도→등급)
python src\trade_area_scoring_snapshot.py || goto :eof

REM 3) 상권 패널(분기별 수요/매출/공급/변화율/변동성/폐업률)
python src\trade_area_panel.py || goto :eof

REM 4) 지도학습 스코어(다음 분기 폐업률 예측 기반 0~100)
python src\trade_area_score_ml.py || goto :eof

REM 5) 리포트(상위 위험 상권 프린트)
python src\report_top_risk.py || goto :eof

REM 6) 폐업률 안정성(평균/표준편차/급변/추세)
python src\analysis_closure_stability.py || goto :eof

REM 7) 요인 분해(ElasticNet/GBR 중요도)
python src\analysis_factor_decomposition.py || goto :eof

REM 8) U-Shape(경쟁/밀도 비선형 관계)
python src\analysis_u_shape.py || goto :eof

REM 9) 공간 확산(Moran's I, Spatial Lag)
python src\analysis_spatial_diffusion.py || goto :eof

echo DONE
endlocal
