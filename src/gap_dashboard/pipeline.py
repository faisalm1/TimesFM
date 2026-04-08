"""Shared ranking pipeline (Streamlit + FastAPI use this)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, List

import pandas as pd

from gap_dashboard.alpaca_daily import load_or_fetch_daily, make_client
from gap_dashboard.config import cache_dir
from gap_dashboard.gap_math import add_overnight_gap_columns, gap_series_for_timesfm
from gap_dashboard.ml_predict import ml_probability_last_bar
from gap_dashboard.timesfm_predict import forecast_gap_decimals, risk_score_down_pct, risk_score_pct


def parse_symbols(text: str) -> List[str]:
    out: List[str] = []
    for line in text.splitlines():
        for part in line.replace(";", ",").split(","):
            s = part.strip().upper()
            if s:
                out.append(s)
    seen = set()
    uniq: List[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


@dataclass
class RankingParams:
    symbols: List[str]
    years: float
    gap_threshold_pct: float
    forward_trading_days: int
    max_context_days: int


@dataclass
class RankingResult:
    rows: List[dict[str, Any]]
    errors: List[dict[str, str]]


def run_ranking(params: RankingParams) -> RankingResult:
    """Load bars, forecast gaps, rank. Alpaca keys read from env inside make_client only."""
    end = date.today()
    start = end - timedelta(days=int(365.25 * params.years))
    cdir = cache_dir()
    client = make_client()

    ok_rows: list = []
    err_rows: list = []
    ranking_at = datetime.now().isoformat(timespec="seconds")

    for sym in params.symbols:
        try:
            df = load_or_fetch_daily(sym, start, end, cdir, client=client)
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = add_overnight_gap_columns(df)
            g = gap_series_for_timesfm(df["overnight_gap_pct"])
            if g.size < 32:
                raise ValueError("Not enough rows with valid overnight gaps (need ≥32).")
            fc = forecast_gap_decimals(
                g,
                horizon=int(params.forward_trading_days),
                max_context=int(params.max_context_days),
            )
            score, mx_pt, mx_q = risk_score_pct(fc, float(params.gap_threshold_pct))
            score_dn, mn_pt, mn_q = risk_score_down_pct(fc, float(params.gap_threshold_pct))
            ml_p, ml_skip = ml_probability_last_bar(df)
            last_close = float(df["close"].iloc[-1])
            # Stylized next-session open if the largest point forecast overnight gap occurred from last close.
            implied_target_px = round(last_close * (1.0 + mx_pt / 100.0), 4)
            implied_down_target_px = round(last_close * (1.0 + mn_pt / 100.0), 4)
            row = {
                "symbol": sym,
                "risk_score": score,
                "max_point_gap_next_pct": mx_pt,
                "max_q90_proxy_pct": mx_q,
                "risk_score_down": score_dn,
                "min_point_gap_next_pct": mn_pt,
                "min_q10_proxy_pct": mn_q,
                "context_days": fc.context_used,
                "last_date": str(df["date"].iloc[-1].date()),
                "last_close": round(last_close, 4),
                "implied_target_px": implied_target_px,
                "implied_down_target_px": implied_down_target_px,
                "forward_sessions": int(params.forward_trading_days),
                "ranking_at": ranking_at,
                "ml_probability": round(ml_p, 6) if ml_p is not None else None,
                "ml_skip_reason": ml_skip,
            }
            ok_rows.append(row)
        except Exception as e:
            err_rows.append({"symbol": sym, "error": f"{type(e).__name__}: {e}"})

    if ok_rows:
        good = pd.DataFrame(ok_rows).sort_values("risk_score", ascending=False, na_position="last")
        ok_rows = good.to_dict(orient="records")

    return RankingResult(rows=ok_rows, errors=err_rows)
