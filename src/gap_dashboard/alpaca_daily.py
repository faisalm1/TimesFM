"""Alpaca daily bars with on-disk Parquet cache (backtests stay fast on repeat runs)."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from gap_dashboard.rate_limit import call_with_alpaca_throttle

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:  # pragma: no cover
    StockHistoricalDataClient = None  # type: ignore[misc, assignment]
    StockBarsRequest = None  # type: ignore[misc, assignment]
    TimeFrame = None  # type: ignore[misc, assignment]


def _env_keys() -> tuple[Optional[str], Optional[str]]:
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET_KEY")
    return key, secret


def make_client() -> Optional["StockHistoricalDataClient"]:
    if StockHistoricalDataClient is None:
        return None
    key, secret = _env_keys()
    if not key or not secret:
        return None
    data_url = os.environ.get("APCA_DATA_BASE_URL") or os.environ.get("ALPACA_DATA_BASE_URL")
    kw = {"api_key": key, "secret_key": secret}
    if data_url:
        kw["url_override"] = data_url.strip()
    return StockHistoricalDataClient(**kw)


def cache_path(symbol: str, start: date, end: date, cache_dir: Path) -> Path:
    safe = symbol.upper().replace("/", "_")
    return cache_dir / f"{safe}_1D_{start.isoformat()}_{end.isoformat()}.parquet"


def bars_to_dataframe(raw) -> pd.DataFrame:
    rows = []
    for sym, barset in raw.data.items():
        for b in barset:
            rows.append(
                {
                    "symbol": sym,
                    "date": pd.Timestamp(b.timestamp).tz_convert(None).normalize(),
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def load_or_fetch_daily(
    symbol: str,
    start: date,
    end: date,
    cache_dir: Path,
    client: Optional["StockHistoricalDataClient"] = None,
) -> pd.DataFrame:
    """Load Parquet cache if present; otherwise fetch from Alpaca and write cache."""
    path = cache_path(symbol, start, end, cache_dir)
    if path.exists():
        return pd.read_parquet(path)

    c = client or make_client()
    if c is None or StockBarsRequest is None or TimeFrame is None:
        raise RuntimeError(
            "Missing Alpaca credentials or alpaca-py. Set APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY, install alpaca-py, or populate cache: " + str(path)
        )

    req = StockBarsRequest(
        symbol_or_symbols=symbol.upper(),
        timeframe=TimeFrame.Day,
        start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc),
        end=datetime(end.year, end.month, end.day, tzinfo=timezone.utc),
    )
    raw = call_with_alpaca_throttle(lambda: c.get_stock_bars(req))
    df = bars_to_dataframe(raw)
    if df.empty:
        raise ValueError(f"No daily bars returned for {symbol} between {start} and {end}")
    df_one = df[df["symbol"] == symbol.upper()].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    df_one.to_parquet(path, index=False)
    return df_one
