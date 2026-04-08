"""
Causal tabular features for gap-event classification.

Labels use only realized overnight gaps from Alpaca daily aggregate bars (no synthetic prices).
Features at row index i use only rows 0..i inclusive — no look-ahead.
The label at i is whether max overnight gap over sessions i+1..i+forward_days is >= threshold
(see gap_math.add_forward_max_gap_label). This is not a simulated trade; it is a historical outcome.

Feature groups:
  - Gap statistics (original 5)
  - Price action / candle anatomy (12 new)
  - Volume signals (3 new)
  - Volatility patterns (6 new)
  - Mean reversion / extremes (3 new)
  - Calendar / regime (3 new, weekday kept from v1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_HISTORY_ROWS = 64

FEATURE_ORDER: list[str] = [
    # ── gap statistics ──
    "abs_overnight_gap_t",
    "mean_abs_overnight_gap_20",
    "std_overnight_gap_60",
    "max_abs_overnight_gap_60",
    "frac_abs_gap_ge_5_60",
    # ── price action / candle anatomy ──
    "close_ret_5",
    "close_ret_10",
    "close_ret_20",
    "close_ret_60",
    "close_to_high_pct",
    "close_to_low_pct",
    "intraday_range_pct",
    "intraday_range_ratio_20d",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "high_vs_20d_high",
    "low_vs_20d_low",
    # ── volume signals ──
    "log_volume_t",
    "volume_ratio_vs_5d",
    "volume_ratio_vs_20d_mean",
    "volume_ratio_vs_60d",
    "volume_trend_5d",
    # ── volatility patterns ──
    "atr_14_pct",
    "atr_ratio_5_20",
    "bollinger_width_20",
    "consecutive_gap_same_dir",
    "gap_acceleration_5d",
    "gap_cluster_count_10d",
    # ── mean reversion / extremes ──
    "rsi_14",
    "close_vs_sma20",
    "close_vs_sma50",
    # ── calendar / regime ──
    "weekday",
    "month",
    "days_since_big_gap",
]


def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-12 else 0.0


def _rolling_mean(arr: np.ndarray, idx: int, window: int) -> float:
    seg = arr[max(0, idx - window + 1): idx + 1]
    return float(np.mean(seg)) if seg.size else 0.0


def _rolling_max(arr: np.ndarray, idx: int, window: int) -> float:
    seg = arr[max(0, idx - window + 1): idx + 1]
    return float(np.max(seg)) if seg.size else 0.0


def _rolling_min(arr: np.ndarray, idx: int, window: int) -> float:
    seg = arr[max(0, idx - window + 1): idx + 1]
    return float(np.min(seg)) if seg.size else 0.0


def _rolling_std(arr: np.ndarray, idx: int, window: int) -> float:
    seg = arr[max(0, idx - window + 1): idx + 1]
    return float(np.nanstd(seg, ddof=0)) if seg.size else 0.0


def _return(arr: np.ndarray, idx: int, lookback: int) -> float:
    if idx >= lookback and abs(arr[idx - lookback]) > 1e-12:
        return float(arr[idx] / arr[idx - lookback] - 1.0)
    return 0.0


def feature_vector(df: pd.DataFrame, idx: int) -> np.ndarray:
    """
    Numeric feature vector at end-of-day idx using only df.iloc[: idx + 1].
    df must include columns: date, open, high, low, close, volume, overnight_gap_pct.
    """
    if idx < 0 or idx >= len(df):
        raise IndexError("idx out of range")

    g = df["overnight_gap_pct"].to_numpy(dtype=float, copy=False)
    opn = df["open"].to_numpy(dtype=float, copy=False)
    hi = df["high"].to_numpy(dtype=float, copy=False)
    lo = df["low"].to_numpy(dtype=float, copy=False)
    clo = df["close"].to_numpy(dtype=float, copy=False)
    vol = df["volume"].to_numpy(dtype=float, copy=False)
    dt = pd.to_datetime(df["date"], utc=False)

    c = float(clo[idx])
    o = float(opn[idx])
    h = float(hi[idx])
    l_ = float(lo[idx])
    v = float(vol[idx])

    # ── gap statistics ──
    abs_t = abs(float(g[idx]))
    seg20 = g[max(0, idx - 19): idx + 1]
    mean_abs_20 = float(np.mean(np.abs(seg20)))
    seg60 = g[max(0, idx - 59): idx + 1]
    std_60 = float(np.nanstd(seg60, ddof=0)) if seg60.size else 0.0
    max_abs_60 = float(np.max(np.abs(seg60))) if seg60.size else 0.0
    frac_ge5 = float(np.mean(np.abs(seg60) >= 5.0)) if seg60.size else 0.0

    # ── price action / candle anatomy ──
    close_ret_5 = _return(clo, idx, 5)
    close_ret_10 = _return(clo, idx, 10)
    close_ret_20 = _return(clo, idx, 20)
    close_ret_60 = _return(clo, idx, 60)

    rng = h - l_
    close_to_high_pct = _safe_div(h - c, rng) * 100.0 if rng > 1e-9 else 50.0
    close_to_low_pct = _safe_div(c - l_, rng) * 100.0 if rng > 1e-9 else 50.0
    intraday_range_pct = _safe_div(rng, c) * 100.0

    range_arr = hi[:idx + 1] - lo[:idx + 1]
    range_mean_20 = _rolling_mean(range_arr, idx, 20)
    intraday_range_ratio_20d = _safe_div(rng, range_mean_20)

    body_pct = _safe_div(abs(c - o), c) * 100.0
    top_body = max(c, o)
    bot_body = min(c, o)
    upper_wick_pct = _safe_div(h - top_body, c) * 100.0
    lower_wick_pct = _safe_div(bot_body - l_, c) * 100.0

    high_20 = _rolling_max(hi, idx, 20)
    low_20 = _rolling_min(lo, idx, 20)
    high_vs_20d_high = _safe_div(c, high_20) if high_20 > 0 else 1.0
    low_vs_20d_low = _safe_div(c, low_20) if low_20 > 0 else 1.0

    # ── volume signals ──
    log_vol = float(np.log1p(max(v, 0.0)))
    vol_mean_5 = _rolling_mean(vol, idx, 5)
    vol_mean_20 = _rolling_mean(vol, idx, 20)
    vol_mean_60 = _rolling_mean(vol, idx, 60)
    volume_ratio_vs_5d = _safe_div(v, vol_mean_5)
    volume_ratio_vs_20d = _safe_div(v, vol_mean_20)
    volume_ratio_vs_60d = _safe_div(v, vol_mean_60)

    vol_seg5 = vol[max(0, idx - 4): idx + 1]
    if vol_seg5.size >= 2:
        x = np.arange(vol_seg5.size, dtype=float)
        slope = float(np.polyfit(x, vol_seg5, 1)[0])
        volume_trend_5d = _safe_div(slope, vol_mean_5)
    else:
        volume_trend_5d = 0.0

    # ── volatility patterns ──
    tr = np.empty(idx + 1, dtype=float)
    for j in range(idx + 1):
        day_range = hi[j] - lo[j]
        if j > 0:
            prev_c = clo[j - 1]
            tr[j] = max(day_range, abs(hi[j] - prev_c), abs(lo[j] - prev_c))
        else:
            tr[j] = day_range

    atr_14 = _rolling_mean(tr, idx, 14)
    atr_14_pct = _safe_div(atr_14, c) * 100.0
    atr_5 = _rolling_mean(tr, idx, 5)
    atr_20 = _rolling_mean(tr, idx, 20)
    atr_ratio_5_20 = _safe_div(atr_5, atr_20)

    sma_20_val = _rolling_mean(clo, idx, 20)
    std_20_val = _rolling_std(clo, idx, 20)
    bollinger_width_20 = _safe_div(2.0 * std_20_val, sma_20_val) * 100.0 if sma_20_val > 0 else 0.0

    consec = 0
    if idx > 0:
        direction = 1 if g[idx] > 0 else -1 if g[idx] < 0 else 0
        if direction != 0:
            consec = 1
            for j in range(idx - 1, -1, -1):
                d2 = 1 if g[j] > 0 else -1 if g[j] < 0 else 0
                if d2 == direction:
                    consec += 1
                else:
                    break
    consecutive_gap_same_dir = float(consec)

    gap_seg5 = g[max(0, idx - 4): idx + 1]
    if gap_seg5.size >= 2:
        abs_gaps = np.abs(gap_seg5)
        x = np.arange(abs_gaps.size, dtype=float)
        gap_acceleration_5d = float(np.polyfit(x, abs_gaps, 1)[0])
    else:
        gap_acceleration_5d = 0.0

    gap_seg10 = g[max(0, idx - 9): idx + 1]
    gap_cluster_count_10d = float(np.sum(np.abs(gap_seg10) >= 3.0))

    # ── mean reversion / extremes ──
    deltas = np.diff(clo[max(0, idx - 14): idx + 1])
    if deltas.size >= 1:
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains)) if gains.size else 0.0
        avg_loss = float(np.mean(losses)) if losses.size else 0.0
        if avg_loss > 1e-12:
            rs = avg_gain / avg_loss
            rsi_14 = 100.0 - 100.0 / (1.0 + rs)
        else:
            rsi_14 = 100.0 if avg_gain > 0 else 50.0
    else:
        rsi_14 = 50.0

    close_vs_sma20 = _safe_div(c, sma_20_val) - 1.0 if sma_20_val > 0 else 0.0
    sma_50_val = _rolling_mean(clo, idx, 50)
    close_vs_sma50 = _safe_div(c, sma_50_val) - 1.0 if sma_50_val > 0 else 0.0

    # ── calendar / regime ──
    ts = dt.iloc[idx]
    weekday = float(min(max(int(ts.weekday()), 0), 4))
    month = float(ts.month)

    big_gap_threshold = 5.0
    days_since_big = 0.0
    for j in range(idx, -1, -1):
        if abs(g[j]) >= big_gap_threshold:
            days_since_big = float(idx - j)
            break
    else:
        days_since_big = float(idx + 1)

    vec = np.array(
        [
            # gap statistics
            abs_t, mean_abs_20, std_60, max_abs_60, frac_ge5,
            # price action / candle anatomy
            close_ret_5, close_ret_10, close_ret_20, close_ret_60,
            close_to_high_pct, close_to_low_pct,
            intraday_range_pct, intraday_range_ratio_20d,
            body_pct, upper_wick_pct, lower_wick_pct,
            high_vs_20d_high, low_vs_20d_low,
            # volume signals
            log_vol, volume_ratio_vs_5d, volume_ratio_vs_20d,
            volume_ratio_vs_60d, volume_trend_5d,
            # volatility patterns
            atr_14_pct, atr_ratio_5_20, bollinger_width_20,
            consecutive_gap_same_dir, gap_acceleration_5d, gap_cluster_count_10d,
            # mean reversion / extremes
            rsi_14, close_vs_sma20, close_vs_sma50,
            # calendar / regime
            weekday, month, days_since_big,
        ],
        dtype=np.float64,
    )
    if vec.shape[0] != len(FEATURE_ORDER):
        raise RuntimeError(
            f"FEATURE_ORDER ({len(FEATURE_ORDER)}) out of sync with feature_vector ({vec.shape[0]})"
        )
    return vec


def build_training_matrix(
    df: pd.DataFrame,
    *,
    threshold_pct: float,
    forward_days: int,
    label_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Rows: each idx from MIN_HISTORY_ROWS to len-2 where forward label is finite.
    Returns X (n, n_features), y (n,) binary, indices used.
    """
    col = label_col or f"label_ge_{threshold_pct:g}pct"
    if col not in df.columns:
        raise KeyError(f"Missing label column {col}; run add_forward_max_gap_label first")

    fwd_col = f"fwd_{forward_days}d_max_gap_pct"
    if fwd_col not in df.columns:
        raise KeyError(f"Missing {fwd_col}")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    idx_list: list[int] = []

    for idx in range(MIN_HISTORY_ROWS, len(df)):
        if pd.isna(df[fwd_col].iloc[idx]):
            continue
        if pd.isna(df[col].iloc[idx]):
            continue
        yi = int(bool(df[col].iloc[idx]))
        X_list.append(feature_vector(df, idx))
        y_list.append(yi)
        idx_list.append(idx)

    if not X_list:
        return np.empty((0, len(FEATURE_ORDER))), np.array([], dtype=np.int64), []

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, idx_list
