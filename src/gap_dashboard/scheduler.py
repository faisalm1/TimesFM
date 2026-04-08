"""Nightly auto-refresh: pull bars, re-forecast, retrain ML — every weekday after market close."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from gap_dashboard.config import ROOT, cache_dir

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None
_last_run: dict[str, Any] | None = None
_run_lock = __import__("threading").Lock()
_running = False

NIGHTLY_HOUR = int(os.environ.get("SCHEDULER_HOUR", "16"))
NIGHTLY_MINUTE = int(os.environ.get("SCHEDULER_MINUTE", "30"))

EMAIL_HOUR = int(os.environ.get("ALERT_EMAIL_HOUR", "15"))
EMAIL_MINUTE = int(os.environ.get("ALERT_EMAIL_MINUTE", "45"))

_HISTORY_PATH = ROOT / "data" / "ml" / "nightly_history.jsonl"


def _append_history(record: dict[str, Any]) -> None:
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _read_recent_history(n: int = 7) -> list[dict[str, Any]]:
    if not _HISTORY_PATH.exists():
        return []
    lines = _HISTORY_PATH.read_text(encoding="utf-8").strip().splitlines()
    out: list[dict[str, Any]] = []
    for ln in lines[-n:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


def nightly_job() -> None:
    """Full pipeline: refresh bars → TimesFM → retrain ML → log delta."""
    global _last_run, _running
    with _run_lock:
        if _running:
            logger.warning("nightly_job already running, skipping")
            return
        _running = True

    t0 = time.perf_counter()
    result: dict[str, Any] = {"started_at": datetime.now().isoformat(timespec="seconds")}
    logger.info("=== NIGHTLY JOB START ===")

    try:
        _step_refresh_bars(result)
        _step_timesfm_forecast(result)
        _step_retrain_ml(result)
    except Exception:
        logger.exception("nightly_job fatal error")
        result["fatal_error"] = True
    finally:
        elapsed = time.perf_counter() - t0
        result["elapsed_seconds"] = round(elapsed, 1)
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _last_run = result
        _append_history(result)
        with _run_lock:
            _running = False
        logger.info("=== NIGHTLY JOB DONE in %.1fs ===", elapsed)


def _step_refresh_bars(result: dict[str, Any]) -> None:
    """Pull fresh bars from Alpaca for every symbol in the universe."""
    logger.info("[nightly] step 1/3: refreshing Alpaca bars")
    t0 = time.perf_counter()

    from gap_dashboard.alpaca_daily import load_or_fetch_daily, make_client

    client = make_client()
    if client is None:
        result["bars_error"] = "Alpaca client unavailable (check .env)"
        logger.error(result["bars_error"])
        return

    symbols = _load_symbol_universe()
    end = date.today()
    start = end - timedelta(days=int(365.25 * 3))
    cdir = cache_dir()

    fetched, cached, errors = 0, 0, 0
    for sym in symbols:
        try:
            cp = _cache_path_for(sym, start, end, cdir)
            if cp.exists():
                cached += 1
                continue
            load_or_fetch_daily(sym, start, end, cdir, client=client)
            fetched += 1
        except Exception as e:
            errors += 1
            logger.debug("bar fetch %s: %s", sym, e)

    result["bars_fetched"] = fetched
    result["bars_cached"] = cached
    result["bars_errors"] = errors
    result["bars_seconds"] = round(time.perf_counter() - t0, 1)
    logger.info("[nightly] bars: %d fetched, %d cached, %d errors in %.1fs",
                fetched, cached, errors, result["bars_seconds"])


def _step_timesfm_forecast(result: dict[str, Any]) -> None:
    """Re-run batch_rank.py to rebuild latest_ranking.json with fresh data."""
    logger.info("[nightly] step 2/3: running TimesFM batch forecast")
    t0 = time.perf_counter()

    import subprocess
    import sys

    script = ROOT / "scripts" / "batch_rank.py"
    if not script.is_file():
        result["forecast_error"] = "batch_rank.py not found"
        return

    cp = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    result["forecast_returncode"] = cp.returncode
    result["forecast_seconds"] = round(time.perf_counter() - t0, 1)

    if cp.returncode != 0:
        result["forecast_error"] = cp.stderr[-500:] if cp.stderr else "non-zero exit"
        logger.error("[nightly] forecast failed: %s", result["forecast_error"])
    else:
        for line in (cp.stdout or "").strip().splitlines()[-5:]:
            logger.info("  batch_rank: %s", line)
        logger.info("[nightly] forecast done in %.1fs", result["forecast_seconds"])


def _step_retrain_ml(result: dict[str, Any]) -> None:
    """Retrain LightGBM on all accumulated Parquet data. Back up old model."""
    logger.info("[nightly] step 3/3: retraining ML model")
    t0 = time.perf_counter()

    ml_dir = ROOT / "data" / "ml"
    model_path = ml_dir / "lgbm_gap.pkl"
    metrics_path = ml_dir / "metrics.jsonl"

    old_metrics = _last_ml_metrics(metrics_path)

    import subprocess
    import sys

    script = ROOT / "scripts" / "train_gap_ml.py"
    if not script.is_file():
        result["retrain_error"] = "train_gap_ml.py not found"
        return

    if model_path.exists():
        backup = ml_dir / f"lgbm_gap_backup_{datetime.now():%Y%m%d_%H%M%S}.pkl"
        try:
            import shutil
            shutil.copy2(model_path, backup)
            logger.info("[nightly] backed up model -> %s", backup.name)
        except Exception as e:
            logger.warning("model backup failed: %s", e)

    cp = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    result["retrain_returncode"] = cp.returncode
    result["retrain_seconds"] = round(time.perf_counter() - t0, 1)

    if cp.returncode != 0:
        result["retrain_error"] = cp.stderr[-500:] if cp.stderr else "non-zero exit"
        logger.error("[nightly] retrain failed: %s", result["retrain_error"])
    else:
        for line in (cp.stdout or "").strip().splitlines()[-3:]:
            logger.info("  train_ml: %s", line)

        new_metrics = _last_ml_metrics(metrics_path)
        if old_metrics and new_metrics:
            delta = (new_metrics.get("roc_auc_val", 0) or 0) - (old_metrics.get("roc_auc_val", 0) or 0)
            result["auc_previous"] = old_metrics.get("roc_auc_val")
            result["auc_new"] = new_metrics.get("roc_auc_val")
            result["auc_delta"] = round(delta, 6)
            result["rows_previous"] = old_metrics.get("n_total")
            result["rows_new"] = new_metrics.get("n_total")
            logger.info("[nightly] AUC %.4f -> %.4f (delta %+.4f), rows %s -> %s",
                        result["auc_previous"] or 0, result["auc_new"] or 0,
                        delta, result["rows_previous"], result["rows_new"])
        elif new_metrics:
            result["auc_new"] = new_metrics.get("roc_auc_val")
            result["rows_new"] = new_metrics.get("n_total")
            logger.info("[nightly] first model: AUC=%.4f rows=%s",
                        result["auc_new"] or 0, result["rows_new"])

        logger.info("[nightly] retrain done in %.1fs", result["retrain_seconds"])


def _last_ml_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def _load_symbol_universe() -> list[str]:
    """Read symbol list from batch_rank.py or fall back to latest_ranking.json."""
    ranking_json = ROOT / "data" / "latest_ranking.json"
    if ranking_json.exists():
        try:
            d = json.loads(ranking_json.read_text(encoding="utf-8"))
            rows = d.get("rows") or []
            syms = [r["symbol"] for r in rows if isinstance(r, dict) and r.get("symbol")]
            if syms:
                return syms
        except Exception:
            pass
    return []


def _cache_path_for(symbol: str, start: date, end: date, cdir: Path) -> Path:
    safe = symbol.upper().replace("/", "_")
    return cdir / f"{safe}_1D_{start.isoformat()}_{end.isoformat()}.parquet"


def get_status() -> dict[str, Any]:
    """Return scheduler status for the /api/scheduler/status endpoint."""
    next_nightly = None
    next_email = None
    enabled = _scheduler is not None and _scheduler.running
    if _scheduler and _scheduler.running:
        for job in _scheduler.get_jobs():
            nf = job.next_run_time
            ts = nf.isoformat(timespec="seconds") if nf else None
            if job.id == "nightly_pipeline":
                next_nightly = ts
            elif job.id == "email_alert":
                next_email = ts

    from gap_dashboard.email_alerts import _smtp_config
    email_configured = _smtp_config() is not None

    return {
        "enabled": enabled,
        "schedule": f"weekdays at {NIGHTLY_HOUR:02d}:{NIGHTLY_MINUTE:02d} ET",
        "next_run": next_nightly,
        "currently_running": _running,
        "last_run": _last_run,
        "recent_history": _read_recent_history(7),
        "email_alert": {
            "schedule": f"weekdays at {EMAIL_HOUR:02d}:{EMAIL_MINUTE:02d} ET",
            "next_run": next_email,
            "configured": email_configured,
        },
    }


def email_alert_job() -> None:
    """Send the daily pre-close email with top picks."""
    logger.info("=== EMAIL ALERT JOB START ===")
    try:
        from gap_dashboard.email_alerts import send_alert_email
        result = send_alert_email()
        logger.info("Email alert result: %s", result)
    except Exception:
        logger.exception("email_alert_job failed")


def start_scheduler() -> None:
    """Attach the nightly cron + email alert to the background scheduler. Idempotent."""
    global _scheduler
    if os.environ.get("SCHEDULER_DISABLE", "").strip().lower() in ("1", "true", "yes"):
        logger.info("Scheduler disabled via SCHEDULER_DISABLE env var")
        return
    if _scheduler is not None:
        return

    _scheduler = BackgroundScheduler(timezone="US/Eastern")
    _scheduler.add_job(
        nightly_job,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=NIGHTLY_HOUR,
            minute=NIGHTLY_MINUTE,
            timezone="US/Eastern",
        ),
        id="nightly_pipeline",
        name="Nightly bar refresh + forecast + ML retrain",
        replace_existing=True,
    )
    _scheduler.add_job(
        email_alert_job,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=EMAIL_HOUR,
            minute=EMAIL_MINUTE,
            timezone="US/Eastern",
        ),
        id="email_alert",
        name=f"Daily top-picks email at {EMAIL_HOUR:02d}:{EMAIL_MINUTE:02d} ET",
        replace_existing=True,
    )
    _scheduler.start()
    for job in _scheduler.get_jobs():
        logger.info("Scheduled: %s — next: %s", job.name, job.next_run_time)


def trigger_now() -> dict[str, str]:
    """Manually trigger an immediate nightly run (non-blocking)."""
    if _running:
        return {"status": "already_running"}
    import threading
    threading.Thread(target=nightly_job, daemon=True, name="nightly-manual").start()
    return {"status": "started"}
