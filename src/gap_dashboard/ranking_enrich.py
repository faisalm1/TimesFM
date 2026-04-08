"""Fill missing fields in latest_ranking.json using Parquet cache + optional TimesFM (real bars only)."""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from gap_dashboard.config import cache_dir
from gap_dashboard.gap_math import add_overnight_gap_columns, gap_series_for_timesfm

logger = logging.getLogger(__name__)


def _safe_symbol_file_prefix(symbol: str) -> str:
    return symbol.upper().replace("/", "_")


def load_symbol_parquet(symbol: str, cdir: Path) -> pd.DataFrame | None:
    """Most recently modified daily Parquet for this symbol in cache."""
    pat = f"{_safe_symbol_file_prefix(symbol)}_1D_*.parquet"
    files = list(cdir.glob(pat))
    if not files:
        return None
    path = max(files, key=lambda p: p.stat().st_mtime)
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.warning("read_parquet failed %s: %s", path, e)
        return None
    if df.empty or "close" not in df.columns:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def _light_enrich_row(row: dict[str, Any], df: pd.DataFrame | None, params: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if df is None or df.empty:
        return out
    last_close = float(df["close"].iloc[-1])
    if out.get("last_close") is None:
        out["last_close"] = round(last_close, 4)
    mx = out.get("max_point_gap_next_pct")
    if mx is not None and out.get("implied_target_px") is None:
        out["implied_target_px"] = round(last_close * (1.0 + float(mx) / 100.0), 4)
    if out.get("forward_sessions") is None and params.get("forward_trading_days") is not None:
        out["forward_sessions"] = int(params["forward_trading_days"])
    if out.get("ml_probability") is None:
        try:
            from gap_dashboard.ml_predict import ml_probability_last_bar

            df2 = add_overnight_gap_columns(df.copy())
            mp, _ = ml_probability_last_bar(df2)
            if mp is not None:
                out["ml_probability"] = round(mp, 6)
        except Exception as e:
            logger.debug("ml skip %s: %s", out.get("symbol"), e)
    return out


def _timesfm_enrich_row(row: dict[str, Any], df: pd.DataFrame | None, params: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if df is None or len(df) < 32:
        return out
    if out.get("risk_score_down") is not None and out.get("min_point_gap_next_pct") is not None:
        return out
    try:
        from gap_dashboard.timesfm_predict import forecast_gap_decimals, risk_score_down_pct, risk_score_pct

        df2 = add_overnight_gap_columns(df.copy())
        g = gap_series_for_timesfm(df2["overnight_gap_pct"])
        if g.size < 32:
            return out
        fwd = int(params.get("forward_trading_days", 5))
        ctx = int(params.get("max_context_days", 512))
        thr = float(params.get("gap_threshold_pct", 10.0))
        fc = forecast_gap_decimals(g, horizon=fwd, max_context=min(ctx, 512))
        sd, mn, mnq = risk_score_down_pct(fc, thr)
        su, mxp, mxq = risk_score_pct(fc, thr)
        lc = float(out.get("last_close") or df["close"].iloc[-1])
        out["risk_score_down"] = round(float(sd), 4)
        out["min_point_gap_next_pct"] = round(float(mn), 4)
        out["min_q10_proxy_pct"] = round(float(mnq), 4)
        out["implied_down_target_px"] = round(lc * (1.0 + float(mn) / 100.0), 4)
        if out.get("max_point_gap_next_pct") is None:
            out["max_point_gap_next_pct"] = round(float(mxp), 4)
        if out.get("max_q90_proxy_pct") is None:
            out["max_q90_proxy_pct"] = round(float(mxq), 4)
        if out.get("risk_score") is None:
            out["risk_score"] = round(float(su), 4)
        if out.get("context_days") is None:
            out["context_days"] = int(fc.context_used)
        if out.get("implied_target_px") is None and out.get("max_point_gap_next_pct") is not None:
            out["implied_target_px"] = round(lc * (1.0 + float(out["max_point_gap_next_pct"]) / 100.0), 4)
    except Exception as e:
        logger.warning("timesfm enrich %s: %s", out.get("symbol"), e)
    return out


def light_enrich_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Fast path: prices, implied up target, ML from cache — no TimesFM."""
    out = copy.deepcopy(data)
    rows = out.get("rows")
    if not isinstance(rows, list):
        return out
    params = out.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    cdir = cache_dir()
    new_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            new_rows.append(row)
            continue
        sym = row.get("symbol")
        if not sym:
            new_rows.append(row)
            continue
        df = load_symbol_parquet(str(sym), cdir)
        new_rows.append(_light_enrich_row(row, df, params))
    out["rows"] = new_rows
    out["enriched_light_by_api"] = True
    return out


def payload_needs_timesfm(data: dict[str, Any]) -> bool:
    rows = data.get("rows") or []
    if not rows or not isinstance(rows, list):
        return False
    r0 = rows[0]
    if not isinstance(r0, dict):
        return False
    return r0.get("risk_score_down") is None or r0.get("min_point_gap_next_pct") is None


def full_enrich_payload_with_timesfm(data: dict[str, Any]) -> dict[str, Any]:
    """Heavy path: TimesFM per row (uses cache Parquet). Caller should run off the request thread."""
    out = copy.deepcopy(data)
    rows = out.get("rows")
    if not isinstance(rows, list):
        return out
    params = out.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    cdir = cache_dir()
    new_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            new_rows.append(row)
            continue
        sym = row.get("symbol")
        if not sym:
            new_rows.append(row)
            continue
        df = load_symbol_parquet(str(sym), cdir)
        r = _light_enrich_row(row, df, params)
        r = _timesfm_enrich_row(r, df, params)
        new_rows.append(r)
    out["rows"] = new_rows
    out["batch_schema_version"] = 2
    out["enriched_full_by_api"] = True
    if "enriched_light_by_api" in out:
        del out["enriched_light_by_api"]
    return out


def write_ranking_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_background_full_enrich(json_path: Path) -> None:
    """Load JSON, run TimesFM for all rows, write back. Log duration."""
    import time

    t0 = time.perf_counter()
    try:
        raw = json_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        data = full_enrich_payload_with_timesfm(data)
        write_ranking_atomic(json_path, data)
        logger.info(
            "ranking full enrich wrote %s rows in %.1fs",
            len(data.get("rows") or []),
            time.perf_counter() - t0,
        )
    except Exception:
        logger.exception("background full enrich failed")
