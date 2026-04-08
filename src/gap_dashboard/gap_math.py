"""Overnight gap: prior regular-session close -> next session open."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_overnight_gap_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Expects columns: open, close; index or column 'date' sorted ascending."""
    out = df.sort_values("date").copy()
    out["prev_close"] = out["close"].shift(1)
    out["overnight_gap_pct"] = (out["open"] / out["prev_close"] - 1.0) * 100.0
    return out


def forward_n_day_max_gap_pct(gap_pct: pd.Series, n_days: int) -> pd.Series:
    """Per row t: max overnight gap % over the next n_days trading sessions (t+1 .. t+n)."""
    arr = gap_pct.to_numpy(dtype=float)
    m = len(arr)
    out = np.full(m, np.nan, dtype=float)
    for i in range(m):
        sl = arr[i + 1 : i + 1 + n_days]
        if sl.size == 0:
            continue
        out[i] = float(np.nanmax(sl))
    return pd.Series(out, index=gap_pct.index)


def add_forward_max_gap_label(
    df: pd.DataFrame, gap_col: str, threshold_pct: float, forward_days: int
) -> pd.DataFrame:
    out = df.copy()
    col = f"fwd_{forward_days}d_max_gap_pct"
    out[col] = forward_n_day_max_gap_pct(out[gap_col], forward_days)
    out[f"label_ge_{threshold_pct:g}pct"] = out[col] >= threshold_pct
    return out


def gap_series_for_timesfm(gap_pct: pd.Series) -> np.ndarray:
    """TimesFM expects numeric series; use decimal returns (e.g. 0.05 for 5%)."""
    s = gap_pct.dropna().astype(float) / 100.0
    return s.to_numpy(dtype=np.float32)
