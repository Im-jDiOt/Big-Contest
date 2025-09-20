import os
from typing import Optional, List
import chardet
import pandas as pd


def detect_encoding(path: str, sample_size: int = 200_000) -> str:
    """Detect file encoding using chardet with a small sample for speed.
    Fallback order: utf-8-sig -> cp949 -> euckr -> chardet result.
    """
    # Quick heuristics by extension
    ext = os.path.splitext(path)[1].lower()
    # Try common encodings directly; if fails, chardet
    tried = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    for enc in tried:
        try:
            with open(path, "r", encoding=enc) as f:
                f.read(1024)
            return enc
        except Exception:
            continue
    # chardet sample
    with open(path, "rb") as f:
        raw = f.read(sample_size)
    guess = chardet.detect(raw)
    enc = guess.get("encoding") or "utf-8"
    return enc


def read_csv_smart(path: str, nrows: Optional[int] = None, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Read CSV with best-effort encoding detection and robust options.
    - Handles large files with low_memory=False.
    - Strips BOM if present.
    - Keeps object dtypes for mixed cols.
    """
    enc = detect_encoding(path)
    try:
        df = pd.read_csv(
            path,
            encoding=enc,
            nrows=nrows,
            usecols=usecols,
            low_memory=False,
        )
        return df
    except UnicodeDecodeError:
        # Fallback attempts
        for enc_try in ["cp949", "euc-kr", "utf-8", "utf-8-sig"]:
            try:
                return pd.read_csv(path, encoding=enc_try, nrows=nrows, usecols=usecols, low_memory=False)
            except Exception:
                continue
        raise


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first column matching any candidate (case-insensitive, substring allowed)."""
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    # exact lower match
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    # substring scan
    for c in candidates:
        for col in cols:
            if c.lower() in col.lower():
                return col
    return None

