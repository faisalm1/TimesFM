"""Local FastAPI server: Alpaca + TimesFM stay server-side (.env never sent to React)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from gap_dashboard.config import ROOT
from gap_dashboard import ranking_enrich
from gap_dashboard import scheduler as nightly_scheduler
from gap_dashboard.pipeline import RankingParams, parse_symbols, run_ranking
from gap_dashboard.timesfm_predict import get_timesfm

load_dotenv()

logging_configured = False
if not logging_configured:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")
    logging_configured = True

app = FastAPI(title="Gap dashboard API", version="0.1.0")


@app.on_event("startup")
def _startup_scheduler():
    nightly_scheduler.start_scheduler()

# Full batch re-run (TimesFM + disk write). Local / trusted use; not a multi-tenant job queue.
_rebuild_lock = threading.Lock()
_rebuild_running = False

# One-shot TimesFM enrich of legacy JSON (gap-down columns) — off request thread.
_enrich_lock = threading.Lock()
_enrich_thread: threading.Thread | None = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RankRequest(BaseModel):
    symbols_text: str = Field(..., description="Comma or newline separated tickers")
    years: float = Field(5, ge=1, le=20)
    gap_threshold_pct: float = Field(10.0, ge=0.1, le=50)
    forward_trading_days: int = Field(5, ge=1, le=20)
    max_context_days: int = Field(512, ge=64, le=512)


class RankResponse(BaseModel):
    rows: list
    errors: list


@app.get("/api/health")
def health():
    """Clients use `features` to detect API age (e.g. in-app rebuild before calling POST)."""
    script = ROOT / "scripts" / "batch_rank.py"
    sched = nightly_scheduler.get_status()
    return {
        "ok": True,
        "api_version": "0.4.0",
        "features": {
            "rebuild_ranking": True,
            "batch_script_present": script.is_file(),
            "nightly_scheduler": True,
            "email_alerts": True,
        },
        "scheduler": {
            "enabled": sched["enabled"],
            "schedule": sched["schedule"],
            "next_run": sched["next_run"],
            "currently_running": sched["currently_running"],
        },
        "email_alert": sched.get("email_alert"),
    }


@app.get("/api/scheduler/status")
def scheduler_status():
    """Full scheduler status including recent run history."""
    return nightly_scheduler.get_status()


@app.post("/api/scheduler/trigger")
def scheduler_trigger():
    """Manually trigger the nightly pipeline immediately."""
    return nightly_scheduler.trigger_now()


@app.post("/api/email/test")
def email_test():
    """Send a test alert email right now using current latest_ranking.json."""
    from gap_dashboard.email_alerts import send_alert_email
    return send_alert_email()


@app.get("/api/email/preview")
def email_preview():
    """Return the HTML that would be emailed (for browser preview)."""
    from gap_dashboard.email_alerts import build_email_html, _load_latest, _enrich_top_rows
    from fastapi.responses import HTMLResponse
    data = _load_latest()
    if data is None:
        raise HTTPException(status_code=404, detail="No ranking data")
    data = _enrich_top_rows(data)
    return HTMLResponse(content=build_email_html(data))


@app.post("/api/rank", response_model=RankResponse)
def rank(body: RankRequest):
    symbols = parse_symbols(body.symbols_text)
    if not symbols:
        raise HTTPException(status_code=400, detail="Add at least one symbol.")
    result = run_ranking(
        RankingParams(
            symbols=symbols,
            years=body.years,
            gap_threshold_pct=body.gap_threshold_pct,
            forward_trading_days=body.forward_trading_days,
            max_context_days=body.max_context_days,
        )
    )
    return RankResponse(rows=result.rows, errors=result.errors)


@app.post("/api/warm-model")
def warm_model():
    """Pre-download / load TimesFM checkpoint (optional)."""
    get_timesfm()
    return {"ok": True, "message": "TimesFM loaded."}


@app.get("/api/latest")
def get_latest_ranking():
    """
    Serve data/latest_ranking.json.

    Legacy files missing prices / ML / gap-down get:
    - immediate light enrich (Parquet cache + ML) in the response;
    - a background job that runs TimesFM for every row and rewrites the JSON (may take many minutes).

    Response extras (not stored in the JSON file):
    - enrich_status: complete | full_pending | light | prices_incomplete
    - rebuild_ranking_running: whether scripts/batch_rank.py is running
    - pending_timesfm_enrich: True only when this request started the TimesFM enrich thread
    """
    p = ROOT / "data" / "latest_ranking.json"
    if not p.exists():
        raise HTTPException(
            status_code=404,
            detail="No saved batch yet. Start a refresh from the app or run a first batch once.",
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    data = ranking_enrich.light_enrich_payload(data)

    need_tf = ranking_enrich.payload_needs_timesfm(data)
    with _rebuild_lock:
        rebuild_busy = _rebuild_running

    global _enrich_thread
    started_bg = False
    if os.environ.get("OVERNIGHT_DISABLE_BG_ENRICH", "").strip().lower() not in ("1", "true", "yes"):
        with _enrich_lock:
            alive = _enrich_thread is not None and _enrich_thread.is_alive()
            if need_tf and not alive and not rebuild_busy:
                _enrich_thread = threading.Thread(
                    target=ranking_enrich.run_background_full_enrich,
                    args=(p,),
                    daemon=True,
                    name="ranking-full-enrich",
                )
                _enrich_thread.start()
                started_bg = True

    rows = data.get("rows") or []
    r0 = rows[0] if rows and isinstance(rows[0], dict) else {}
    with _enrich_lock:
        enrich_thread_alive = _enrich_thread is not None and _enrich_thread.is_alive()

    if r0.get("last_close") is None:
        enrich_status = "prices_incomplete"
    elif not need_tf:
        enrich_status = "complete"
    elif enrich_thread_alive or started_bg:
        enrich_status = "full_pending"
    else:
        enrich_status = "light"

    data["enrich_status"] = enrich_status
    data["rebuild_ranking_running"] = rebuild_busy
    if started_bg:
        data["pending_timesfm_enrich"] = True

    return data


def _run_batch_rank_script() -> None:
    """Execute scripts/batch_rank.py with project root as cwd (same as manual CLI)."""
    script = ROOT / "scripts" / "batch_rank.py"
    if not script.is_file():
        return
    subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        env=os.environ.copy(),
    )


@app.get("/api/rebuild-ranking/status")
def rebuild_ranking_status():
    """Whether a background full-batch refresh is in progress."""
    with _rebuild_lock:
        return {"running": _rebuild_running}


@app.post("/api/rebuild-ranking")
def rebuild_ranking():
    """
    Rebuild data/latest_ranking.json in a background thread (same work as scripts/batch_rank.py).
    Returns 202 immediately; may take many minutes. 409 if already running.
    """
    global _rebuild_running
    script = ROOT / "scripts" / "batch_rank.py"
    if not script.is_file():
        raise HTTPException(status_code=500, detail="batch_rank.py not found in project.")

    with _rebuild_lock:
        if _rebuild_running:
            raise HTTPException(status_code=409, detail="A refresh is already in progress.")
        _rebuild_running = True

    def _job() -> None:
        global _rebuild_running
        try:
            _run_batch_rank_script()
        finally:
            with _rebuild_lock:
                _rebuild_running = False

    threading.Thread(target=_job, daemon=True).start()
    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "message": "Refreshing rankings in the background. This may take several minutes.",
        },
    )
