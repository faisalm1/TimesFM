"""Daily pre-close email: top 5 gap-up + top 5 gap-down picks from latest_ranking.json."""

from __future__ import annotations

import json
import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from gap_dashboard.config import ROOT

logger = logging.getLogger(__name__)

TOP_N = 5


def _smtp_config() -> dict[str, Any] | None:
    to = os.environ.get("ALERT_EMAIL_TO", "").strip()
    frm = os.environ.get("ALERT_EMAIL_FROM", "").strip()
    host = os.environ.get("ALERT_SMTP_HOST", "smtp.gmail.com").strip()
    port = int(os.environ.get("ALERT_SMTP_PORT", "587"))
    pwd = os.environ.get("ALERT_SMTP_PASSWORD", "").strip()
    if not to or not frm or not pwd:
        return None
    return {"to": to, "from": frm, "host": host, "port": port, "password": pwd}


def _load_latest() -> dict[str, Any] | None:
    p = ROOT / "data" / "latest_ranking.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "—"
    return f"{float(v):+.2f}%"


def _fmt_dollar(v: Any) -> str:
    if v is None:
        return "—"
    return f"${float(v):,.2f}"


def _fmt_score(v: Any) -> str:
    if v is None:
        return "—"
    return f"{float(v):.1f}"


def _fmt_ml(v: Any) -> str:
    if v is None:
        return "—"
    return f"{float(v):.2%}"


def _pct_to_target(last: Any, tgt: Any) -> str:
    if last is None or tgt is None:
        return "—"
    l, t = float(last), float(tgt)
    if abs(l) < 1e-9:
        return "—"
    return f"{((t / l) - 1) * 100:+.2f}%"


def _row_html_up(r: dict[str, Any], rank: int) -> str:
    sym = r.get("symbol", "?")
    return f"""<tr>
  <td style="padding:6px 10px;font-weight:700;color:#22d3ee">{rank}. {sym}</td>
  <td style="padding:6px 10px;text-align:right">{_fmt_score(r.get("risk_score"))}</td>
  <td style="padding:6px 10px;text-align:right">{_fmt_ml(r.get("ml_probability"))}</td>
  <td style="padding:6px 10px;text-align:right">{_fmt_pct(r.get("max_point_gap_next_pct"))}</td>
  <td style="padding:6px 10px;text-align:right">{_fmt_pct(r.get("max_q90_proxy_pct"))}</td>
  <td style="padding:6px 10px;text-align:right">{_fmt_dollar(r.get("last_close"))}</td>
  <td style="padding:6px 10px;text-align:right">{_fmt_dollar(r.get("implied_target_px"))}</td>
  <td style="padding:6px 10px;text-align:right">{_pct_to_target(r.get("last_close"), r.get("implied_target_px"))}</td>
</tr>"""


def _row_html_down(r: dict[str, Any], rank: int) -> str:
    sym = r.get("symbol", "?")
    return f"""<tr>
  <td style="padding:6px 10px;font-weight:700;color:#f472b6">{rank}. {sym}</td>
  <td style="padding:6px 10px;text-align:right">{_fmt_score(r.get("risk_score_down"))}</td>
  <td style="padding:6px 10px;text-align:right">{_fmt_ml(r.get("ml_probability"))}</td>
  <td style="padding:6px 10px;text-align:right">{_fmt_pct(r.get("min_point_gap_next_pct"))}</td>
  <td style="padding:6px 10px;text-align:right">{_fmt_pct(r.get("min_q10_proxy_pct"))}</td>
  <td style="padding:6px 10px;text-align:right">{_fmt_dollar(r.get("last_close"))}</td>
  <td style="padding:6px 10px;text-align:right">{_fmt_dollar(r.get("implied_down_target_px"))}</td>
  <td style="padding:6px 10px;text-align:right">{_pct_to_target(r.get("last_close"), r.get("implied_down_target_px"))}</td>
</tr>"""


def build_email_html(data: dict[str, Any]) -> str:
    rows = data.get("rows") or []
    params = data.get("params") or {}

    up_sorted = sorted(
        [r for r in rows if r.get("risk_score") is not None],
        key=lambda r: float(r.get("risk_score", 0)),
        reverse=True,
    )[:TOP_N]

    down_sorted = sorted(
        [r for r in rows if r.get("risk_score_down") is not None],
        key=lambda r: float(r.get("risk_score_down", 0)),
        reverse=True,
    )[:TOP_N]

    gen_at = data.get("generated_at", "unknown")
    threshold = params.get("gap_threshold_pct", "?")
    fwd = params.get("forward_trading_days", "?")
    now_str = datetime.now().strftime("%B %d, %Y %I:%M %p ET")

    th_style = 'style="padding:6px 10px;text-align:right;border-bottom:2px solid #334155;color:#94a3b8;font-size:12px"'
    th_left = 'style="padding:6px 10px;text-align:left;border-bottom:2px solid #334155;color:#94a3b8;font-size:12px"'

    up_rows = "\n".join(_row_html_up(r, i + 1) for i, r in enumerate(up_sorted))
    down_rows = "\n".join(_row_html_down(r, i + 1) for i, r in enumerate(down_sorted))

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#0f172a;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif">
<div style="max-width:720px;margin:0 auto;padding:24px 16px">

<div style="text-align:center;margin-bottom:24px">
  <h1 style="color:#f8fafc;font-size:22px;margin:0 0 4px">Overnight Lab — Daily Alert</h1>
  <p style="color:#94a3b8;font-size:13px;margin:0">{now_str} &nbsp;|&nbsp; {threshold}% threshold &nbsp;|&nbsp; {fwd}-session horizon</p>
  <p style="color:#64748b;font-size:11px;margin:4px 0 0">Data scored: {gen_at}</p>
</div>

<!-- GAP UP -->
<div style="margin-bottom:28px">
  <h2 style="color:#22d3ee;font-size:16px;margin:0 0 8px">&#9650; Top {TOP_N} Gap-Up Candidates</h2>
  <table style="width:100%;border-collapse:collapse;background:#1e293b;border-radius:8px;overflow:hidden;font-size:13px">
    <thead><tr>
      <th {th_left}>Symbol</th>
      <th {th_style}>Risk</th>
      <th {th_style}>ML P</th>
      <th {th_style}>Max Pt%</th>
      <th {th_style}>Q90%</th>
      <th {th_style}>Last $</th>
      <th {th_style}>Target $</th>
      <th {th_style}>% to Tgt</th>
    </tr></thead>
    <tbody>
{up_rows}
    </tbody>
  </table>
</div>

<!-- GAP DOWN -->
<div style="margin-bottom:28px">
  <h2 style="color:#f472b6;font-size:16px;margin:0 0 8px">&#9660; Top {TOP_N} Gap-Down Candidates</h2>
  <table style="width:100%;border-collapse:collapse;background:#1e293b;border-radius:8px;overflow:hidden;font-size:13px">
    <thead><tr>
      <th {th_left}>Symbol</th>
      <th {th_style}>Risk</th>
      <th {th_style}>ML P</th>
      <th {th_style}>Min Pt%</th>
      <th {th_style}>Q10%</th>
      <th {th_style}>Last $</th>
      <th {th_style}>Target $</th>
      <th {th_style}>% to Tgt</th>
    </tr></thead>
    <tbody>
{down_rows}
    </tbody>
  </table>
</div>

<!-- LEGEND -->
<div style="background:#1e293b;border-radius:8px;padding:14px 16px;font-size:11px;color:#94a3b8;line-height:1.7">
  <strong style="color:#cbd5e1">Column guide</strong><br>
  <b>Risk</b> — TimesFM-derived 0–100 score (tail magnitude vs threshold).<br>
  <b>ML P</b> — LightGBM probability of a &ge;{threshold}% gap within {fwd} sessions (35 causal features, daily retrained).<br>
  <b>Max/Min Pt%</b> — largest positive / most negative forecast gap in the horizon.<br>
  <b>Q90/Q10%</b> — upper/lower quantile band (uncalibrated).<br>
  <b>Target $</b> — implied next open if the max/min point gap is realized.<br>
  <b>% to Tgt</b> — distance from last close to target price.
</div>

<p style="text-align:center;color:#475569;font-size:10px;margin-top:20px">
  Not financial advice. Forecasts are experimental — validate out-of-sample before production use.<br>
  Overnight Lab &mdash; automated by nightly scheduler.
</p>

</div>
</body>
</html>"""
    return html


def _enrich_top_rows(data: dict[str, Any]) -> dict[str, Any]:
    """Light-enrich only the rows that might appear in the email (top N by each board)."""
    try:
        from gap_dashboard.ranking_enrich import load_symbol_parquet, _light_enrich_row
        from gap_dashboard.config import cache_dir
    except Exception:
        logger.warning("Cannot import enrichment modules; skipping ML fill")
        return data

    rows = data.get("rows") or []
    params = data.get("params") or {}
    cdir = cache_dir()

    up_pool = sorted(
        [r for r in rows if r.get("risk_score") is not None],
        key=lambda r: float(r.get("risk_score", 0)), reverse=True,
    )[:TOP_N]
    down_pool = sorted(
        [r for r in rows if r.get("risk_score_down") is not None],
        key=lambda r: float(r.get("risk_score_down", 0)), reverse=True,
    )[:TOP_N]

    need_enrich = {r.get("symbol") for r in up_pool + down_pool if r.get("symbol")}
    enriched: dict[str, dict] = {}
    for sym in need_enrich:
        row = next((r for r in rows if r.get("symbol") == sym), None)
        if row is None:
            continue
        df = load_symbol_parquet(sym, cdir)
        enriched[sym] = _light_enrich_row(row, df, params)
        logger.info("email enrich %s → ml_probability=%s", sym, enriched[sym].get("ml_probability"))

    new_rows = [enriched.get(r.get("symbol"), r) if r.get("symbol") in enriched else r for r in rows]
    out = dict(data, rows=new_rows)
    return out


def send_alert_email(data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build and send the daily alert. Returns status dict."""
    cfg = _smtp_config()
    if cfg is None:
        return {"sent": False, "reason": "SMTP not configured (set ALERT_EMAIL_TO, ALERT_EMAIL_FROM, ALERT_SMTP_PASSWORD)"}

    if data is None:
        data = _load_latest()
    if data is None:
        return {"sent": False, "reason": "No latest_ranking.json found"}
    data = _enrich_top_rows(data)

    rows = data.get("rows") or []
    if not rows:
        return {"sent": False, "reason": "No rows in ranking data"}

    html = build_email_html(data)
    now_str = datetime.now().strftime("%b %d %Y")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Overnight Lab — Top Picks for {now_str}"
    msg["From"] = cfg["from"]
    msg["To"] = cfg["to"]

    plain = (
        f"Overnight Lab Daily Alert — {now_str}\n\n"
        "Open this email in an HTML-capable client to see the full tables.\n"
        "Or check your dashboard at http://localhost:5173/"
    )
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as srv:
            srv.ehlo()
            srv.starttls()
            srv.ehlo()
            srv.login(cfg["from"], cfg["password"])
            srv.sendmail(cfg["from"], [cfg["to"]], msg.as_string())
        logger.info("Alert email sent to %s", cfg["to"])
        return {"sent": True, "to": cfg["to"], "subject": msg["Subject"]}
    except Exception as e:
        logger.exception("Failed to send alert email")
        return {"sent": False, "reason": str(e)}
