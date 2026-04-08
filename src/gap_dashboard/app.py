"""Streamlit dashboard (optional). Primary UI is React + FastAPI — see README."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from gap_dashboard.alpaca_daily import make_client
from gap_dashboard.config import (
    DEFAULT_CONTEXT_DAYS,
    DEFAULT_FORWARD_DAYS,
    DEFAULT_GAP_THRESHOLD_PCT,
    DEFAULT_HISTORY_YEARS,
    cache_dir,
)
from gap_dashboard.pipeline import RankingParams, parse_symbols, run_ranking
from gap_dashboard.timesfm_predict import get_timesfm

load_dotenv()

st.set_page_config(page_title="Overnight gap risk (TimesFM)", layout="wide")
st.title("US equities — overnight gap risk (TimesFM)")
st.caption(
    "Gap = prior regular-session **close** → next session **open**. "
    "Prefer the **React** UI (`web/` + FastAPI) for local dev."
)


@st.cache_resource(show_spinner="Loading TimesFM (first run may download the checkpoint)…")
def _warm_timesfm():
    return get_timesfm()


with st.sidebar:
    st.header("Data")
    sym_text = st.text_area(
        "Symbols (comma or newline)",
        value="AAPL\nMSFT\nNVDA",
        height=120,
    )
    years = st.number_input("Years of daily history", min_value=1, max_value=20, value=DEFAULT_HISTORY_YEARS)
    thr = st.number_input("Gap threshold (%)", min_value=0.1, max_value=50.0, value=DEFAULT_GAP_THRESHOLD_PCT, step=0.5)
    fwd = st.number_input("Forward window (trading days)", min_value=1, max_value=20, value=DEFAULT_FORWARD_DAYS)
    ctx = st.number_input("TimesFM max context (days)", min_value=64, max_value=512, value=min(DEFAULT_CONTEXT_DAYS, 512), step=32)

    st.header("Model")
    if st.button("Preload TimesFM"):
        _warm_timesfm()
        st.success("TimesFM ready.")

    run = st.button("Run ranking", type="primary")

cdir = cache_dir()
st.sidebar.caption(f"Parquet cache: `{cdir}`")

if run:
    symbols = parse_symbols(sym_text)
    if not symbols:
        st.error("Add at least one symbol.")
    else:
        end = date.today()
        start = end - timedelta(days=int(365.25 * years))
        client = make_client()
        if client is None:
            st.warning(
                "Alpaca keys not set (`APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`). "
                "Fetches only work if Parquet cache already exists for that symbol/range."
            )

        result = run_ranking(
            RankingParams(
                symbols=symbols,
                years=float(years),
                gap_threshold_pct=float(thr),
                forward_trading_days=int(fwd),
                max_context_days=int(ctx),
            )
        )
        good = pd.DataFrame(result.rows)
        err_df = pd.DataFrame(result.errors)

        if not good.empty:
            good = good.sort_values("risk_score", ascending=False, na_position="last")
        st.subheader("Ranked by heuristic risk score (higher = larger model-implied gap tail vs threshold)")
        st.dataframe(good if not good.empty else pd.DataFrame(columns=["symbol", "risk_score"]), use_container_width=True, hide_index=True)

        if not err_df.empty:
            with st.expander("Symbols with errors"):
                st.dataframe(err_df, use_container_width=True)

        st.caption(
            "TimesFM quantile heads are experimental/uncalibrated. "
            "Validate on your own walk-forward tests before trading."
        )

else:
    st.info(
        "Use the **React** app in `web/` with the FastAPI server for the main UI. "
        "Configure symbols in the sidebar here if you still use Streamlit."
    )
