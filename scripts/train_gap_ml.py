"""
Train LightGBM on causal features vs realized forward-gap labels (Alpaca daily bars only).

No simulated fills: labels are max overnight gap % over the next N sessions from real OHLC.
Walk-forward style split: rows sorted by as-of date, last fraction held out for validation.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import numpy as np
import pandas as pd

from gap_dashboard.config import cache_dir
from gap_dashboard.gap_math import add_forward_max_gap_label, add_overnight_gap_columns
from gap_dashboard.ml_features import FEATURE_ORDER, build_training_matrix

try:
    import joblib
    from lightgbm import LGBMClassifier
    from sklearn.metrics import recall_score, roc_auc_score
    from sklearn.metrics import precision_score
except ImportError as e:
    raise SystemExit(
        "Install ML deps: pip install lightgbm scikit-learn joblib\n" + str(e)
    ) from e


VAL_FRACTION = 0.2


def _iter_cache_parquets(cdir: Path) -> list[Path]:
    if not cdir.is_dir():
        return []
    return sorted(cdir.glob("*.parquet"))


def main() -> None:
    t0 = datetime.now()
    print(f"[{t0:%H:%M:%S}] train_gap_ml: scanning cache in {cache_dir()}")

    paths = _iter_cache_parquets(cache_dir())
    if not paths:
        print("No Parquet files in cache. Run batch fetch / batch_rank first.")
        return

    X_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    date_blocks: list[np.ndarray] = []

    threshold = 10.0
    forward_days = 5

    for p in paths:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        if df.empty or "close" not in df.columns:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = add_overnight_gap_columns(df)
        df = add_forward_max_gap_label(df, "overnight_gap_pct", threshold, forward_days)

        X, y, idx_list = build_training_matrix(
            df, threshold_pct=threshold, forward_days=forward_days
        )
        if X.shape[0] == 0:
            continue
        dt = np.array([df["date"].iloc[i] for i in idx_list], dtype="datetime64[ns]")
        X_blocks.append(X)
        y_blocks.append(y)
        date_blocks.append(dt)

    if not X_blocks:
        print("No training rows built. Need enough history per symbol.")
        return

    X_all = np.vstack(X_blocks)
    y_all = np.concatenate(y_blocks)
    d_all = np.concatenate(date_blocks)

    order = np.argsort(d_all)
    X_all = X_all[order]
    y_all = y_all[order]

    n = X_all.shape[0]
    split = max(1, int(n * (1.0 - VAL_FRACTION)))
    X_tr, X_va = X_all[:split], X_all[split:]
    y_tr, y_va = y_all[:split], y_all[split:]

    pos = float(y_tr.sum())
    neg = float(len(y_tr) - pos)
    print(
        f"[{datetime.now():%H:%M:%S}] rows={n} train={len(y_tr)} val={len(y_va)} "
        f"pos_rate_tr={pos / max(len(y_tr), 1):.4f}"
    )

    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=48,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        class_weight="balanced",
        verbose=-1,
    )
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_va)[:, 1]
    pred = (proba >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_va, proba)) if len(np.unique(y_va)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    prec = float(precision_score(y_va, pred, zero_division=0))
    rec = float(recall_score(y_va, pred, zero_division=0))

    out_dir = Path(__file__).resolve().parents[1] / "data" / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_p = out_dir / "lgbm_gap.pkl"
    feat_p = out_dir / "features.json"
    metrics_p = out_dir / "metrics.jsonl"

    joblib.dump(clf, model_p)
    feat_p.write_text(json.dumps(FEATURE_ORDER, indent=2), encoding="utf-8")

    row = {
        "timestamp": datetime.now().isoformat(),
        "n_total": n,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "roc_auc_val": auc,
        "precision_val_0p5": prec,
        "recall_val_0p5": rec,
        "label": f">={threshold}% max overnight gap within {forward_days} sessions (realized)",
        "features": "causal OHLCV/gap stats only; no lookahead",
    }
    with metrics_p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    print(f"[{datetime.now():%H:%M:%S}] saved model -> {model_p}")
    print(f"  val ROC-AUC={auc:.4f}  precision@0.5={prec:.4f}  recall@0.5={rec:.4f}")
    print(f"  metrics appended to {metrics_p}")


if __name__ == "__main__":
    main()
