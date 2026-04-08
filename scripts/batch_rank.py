"""Batch rank ~700 TC2000 tickers: fetch 3yr daily bars, run TimesFM, write JSON."""

from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd

from gap_dashboard.alpaca_daily import load_or_fetch_daily, make_client
from gap_dashboard.config import cache_dir
from gap_dashboard.gap_math import add_overnight_gap_columns, gap_series_for_timesfm
from gap_dashboard.ml_predict import ml_probability_last_bar
from gap_dashboard.timesfm_predict import forecast_gap_decimals, get_timesfm, risk_score_down_pct, risk_score_pct

SYMBOLS_RAW = """
CAR UVIX PSKY UNH HTZ UVXY MSTZ HUM VXX STUB CVS AAOI AVGO BOIL ANET ZSL WULF CRWD AXTI OCUL
LITE RIOT SQQQ CF VERU PANW HUT CRWV UNG CRDO LW LABD SMMT VLO SPXS INTC SPXU DELL HAL APT
ZM CIFR RBRK CIEN AA CNQ USO VG CNC NBIS DOW PAA WMB TZA EOG NTR CORZ SOXS STX OSCR EPD CTRA
CLF GTLB GME DVN CVX ALT CRCL CLOV AR ET TMUS SU RIG WOLF SIRI CHTR TLN CC VOD SNAP HOG XOP
MO VST BTG CDE GOOG CME MRNA GDDY RDDT SMCI MGM GOOGL LYFT APA NRG LEVI PGR BIIB SLB TWLO XLE
WDC FTNT EMR KMI SOC LUMN NEM LNG MELI JBL EQT FUBO NEE SRPT MDT DKNG DBRG GLW OXY MS CI PLTR
NET CAVA IAU GLD CLS HL SHEL GEV AES WPM RXRX ROKU JPM BX BEKE NVAX DDOG BRKB GLXY MPT PNC IEF
SO ANF AFL COP XLU ACI IOVA FLNA PTON IQ BROS BKLN ETN FLY CSCO LLY LQD DBX XLV BAC CWAN RUN
GDX KRE NVO MT TTWO CAH CRSP OWL KR IRM MNKD USB WBD LVS DXCM CMCSA HYG GDXJ ABT IYR GEMI EA
DLR SONY TLT UNP RBLX HSBC TER BP BILL ODD UPS AXP PINS KGC VALE SCHW XLF AMC CAT CART ARES XOM
M TMO AMZN DASH ALB ADP GE NUGT B SCHD DE SMH LRCX SOXX XLC HPE TXN SBUX NFLX PCG BBY WFC IWM
AMBA BAX TSM AEM V HSY BLK ZIM FDX HWM IREN ISRG RSP CZR ZS FSLY U VOO IVV ASHR SPY XLK MMM XBI
FHN C PM IGV DPST SGML SNPS EEM CVNA DIA AKAM RTX GS SIG FUTU ETSY MOS ANVS XLI PBR CDNS SNOW
MCHP AMGN XLB DIS CSX BLSH AIG ZETA SILJ TMF VRT NLY BTDR APO PAAS VRTX NUAI NOK CRH ACN ON QQQ
MSFT PATH SLS RNG KKR VFC SPOT EFA FEZ STZ ACMR VZ URI JNJ DHR EWY ABBV MA MDB UBER COST FXI BULL
TSEM META AMD NVDA ICLN AAP ASML CAG AEO JD CRM TRIP BSX EWZ COHR FIS RUM XRT ADI T ADSK CCJ EWJ
APP REGN AMAT MDLZ PL PG BYND CTAS KWEB AGNC SPGI MAGS COPX SAP ARKK BE S F WGS FCX IBM FAS GSK
BIDU GILD SSO ARKG LYV GD CL MRVL NMAX ALAB CPNG LI APLD AI GEHC SOFI KHC HRL EBAY GTM SOXL ASAN
DOCU SHAK XYZ TECK SKYT CHWY IBKR TDOC BUD LHX MCD VKTX PYPL OPAD WDAY MARA IP AG LMT TNA COF
HPQ XLP DECK FBTC GBTC RACE STLA OUST GRAB MU IBIT KLAR ULTA BITO TEVA COIN PDD PSX LULU SLV FIG
LUV BW ADBE WRBY KO TXRH SPXL GAP CHPT ONDS XLY DOCN SEDG INTU UPRO TMQ BILI ABR BBAI PPG EXPE
LQDA METU QCOM XPEV GM TE LUNR NVDL SE LAC FSLR CMG JETS BBWI ALDX UPST NVDX BKNG MRK AMDL BA
PEP FISV HOOD ORCL PFE UAMY QXO LABU DAL NU OPEN BB UMAC LOW HLT WYNN URA CEG OKTA ABNB TEM ABTC
ROST TJX HIVE INOD SATS NOW EL AVAV ETH URBN NIO ETHA LASR BMY GOOS NFE PCT SCCO SOUN ETHE AAL
ALGN HON DJT TGT TQQQ SHOP CLSK XHB DG KSS MBLY XP DDD ASPI FTAI AAPL AFRM SG PSQH BKKT TLRY
QS QSI JBLU WEN MAR RDW ATYR ABVX QUBT RANI JMIA TPR BHC BABA DUOL ABAT HD GRRR SCO MP RILY YINN
TSLA TOST UEC TIGR GNRC CGC IONQ KVUE POET NKE IOT RH UUUU CCL RGTI BTU PONY WMT CONL AGQ UAL
CRML TSSI BITX HLF ELF ASTS BURL RCL QBTS DHI WWR KTOS DNN CPB W ASO SBET IBRX FIGR BMNR LEN
FRMI SPCE AUR BTBT DAVE ARM HIMS MSTR SYM KMB TSCO SNDK KOPN PHM ONON DNUT KOLD TEAM BRR UWMC
NVTS NCLH ACHR LDI RIVN DLTR RR RKLB CELH DFDV RKT FFAI JOBY MVIS SERV TMC USAR CSIQ ETHU RXT
OKLO FCEL KEEL ENPH RCAT SVIX ENVX NAK TSLL INO LCID LMND TTD UPXI ORBS NN LASE NNE REPL EOSE NB
MSOS ASST PLUG MSTX RZLV MSTU SMR NAIL AXON LAES
"""

YEARS = 3
GAP_THRESHOLD_PCT = 10.0
FORWARD_TRADING_DAYS = 5
MAX_CONTEXT_DAYS = 512


def main():
    symbols = list(dict.fromkeys(s for s in SYMBOLS_RAW.split() if s))
    print(f"[{datetime.now():%H:%M:%S}] {len(symbols)} symbols to process (3yr, threshold {GAP_THRESHOLD_PCT}%)")

    end = date.today()
    start = end - timedelta(days=int(365.25 * YEARS))
    cdir = cache_dir()
    client = make_client()
    if client is None:
        print("ERROR: Alpaca client is None. Check .env keys.")
        return

    print(f"[{datetime.now():%H:%M:%S}] Phase 1: fetching daily bars (cached after first run)…")
    bar_data: dict[str, pd.DataFrame] = {}
    fetch_errors: list[dict] = []
    for i, sym in enumerate(symbols):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{datetime.now():%H:%M:%S}] fetching {i+1}/{len(symbols)}: {sym}")
        try:
            df = load_or_fetch_daily(sym, start, end, cdir, client=client)
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            bar_data[sym] = df
        except Exception as e:
            fetch_errors.append({"symbol": sym, "error": f"{type(e).__name__}: {e}"})
    print(f"[{datetime.now():%H:%M:%S}] Bars: {len(bar_data)} ok, {len(fetch_errors)} errors")

    print(f"[{datetime.now():%H:%M:%S}] Phase 2: loading TimesFM…")
    get_timesfm()
    print(f"[{datetime.now():%H:%M:%S}] TimesFM ready")

    print(f"[{datetime.now():%H:%M:%S}] Phase 3: running predictions…")
    ranking_at = datetime.now().isoformat(timespec="seconds")
    ok_rows: list[dict] = []
    pred_errors: list[dict] = []
    t0 = time.time()
    for i, (sym, df) in enumerate(bar_data.items()):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  [{datetime.now():%H:%M:%S}] predicting {i+1}/{len(bar_data)}: {sym}  (elapsed {elapsed:.0f}s)")
        try:
            df2 = add_overnight_gap_columns(df)
            g = gap_series_for_timesfm(df2["overnight_gap_pct"])
            if g.size < 32:
                raise ValueError("Not enough gap observations.")
            fc = forecast_gap_decimals(g, horizon=FORWARD_TRADING_DAYS, max_context=MAX_CONTEXT_DAYS)
            score, mx_pt, mx_q = risk_score_pct(fc, GAP_THRESHOLD_PCT)
            score_dn, mn_pt, mn_q = risk_score_down_pct(fc, GAP_THRESHOLD_PCT)
            ml_p, ml_skip = ml_probability_last_bar(df2)
            last_close = float(df2["close"].iloc[-1])
            implied_target_px = round(last_close * (1.0 + mx_pt / 100.0), 4)
            implied_down_target_px = round(last_close * (1.0 + mn_pt / 100.0), 4)
            rec = {
                "symbol": sym,
                "risk_score": round(score, 4),
                "max_point_gap_next_pct": round(mx_pt, 4),
                "max_q90_proxy_pct": round(mx_q, 4),
                "risk_score_down": round(score_dn, 4),
                "min_point_gap_next_pct": round(mn_pt, 4),
                "min_q10_proxy_pct": round(mn_q, 4),
                "context_days": fc.context_used,
                "last_date": str(df2["date"].iloc[-1].date()),
                "last_close": round(last_close, 4),
                "implied_target_px": implied_target_px,
                "implied_down_target_px": implied_down_target_px,
                "forward_sessions": FORWARD_TRADING_DAYS,
                "ranking_at": ranking_at,
            }
            if ml_p is not None:
                rec["ml_probability"] = round(ml_p, 6)
            if ml_skip:
                rec["ml_skip_reason"] = ml_skip
            ok_rows.append(rec)
        except Exception as e:
            pred_errors.append({"symbol": sym, "error": f"{type(e).__name__}: {e}"})

    ok_rows.sort(key=lambda r: r["risk_score"], reverse=True)
    all_errors = fetch_errors + pred_errors

    out_path = Path(__file__).resolve().parents[1] / "data" / "latest_ranking.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_schema_version": 2,
        "generated_at": ranking_at,
        "params": {
            "years": YEARS,
            "gap_threshold_pct": GAP_THRESHOLD_PCT,
            "forward_trading_days": FORWARD_TRADING_DAYS,
            "max_context_days": MAX_CONTEXT_DAYS,
        },
        "total_symbols": len(symbols),
        "rows": ok_rows,
        "errors": all_errors,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[{datetime.now():%H:%M:%S}] Done. {len(ok_rows)} ranked, {len(all_errors)} errors.")
    print(f"  Top 20:")
    for r in ok_rows[:20]:
        print(f"    {r['symbol']:8s}  score={r['risk_score']:7.2f}  pt_gap={r['max_point_gap_next_pct']:7.2f}%  q90={r['max_q90_proxy_pct']:7.2f}%")
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
