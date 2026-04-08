"""TimesFM forecasts on overnight gap series (close->next open, as decimals)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import timesfm as _tfm
except ImportError:  # pragma: no cover
    _tfm = None  # type: ignore[assignment]


@dataclass
class GapForecastResult:
    point: np.ndarray  # (horizon,)
    quantiles: np.ndarray  # (horizon, n_q) mean strip excluded
    context_used: int
    horizon: int


_tfm_model = None


def _repo_id() -> str:
    return os.environ.get("TIMESFM_HF_REPO", "google/timesfm-1.0-200m-pytorch")


def get_timesfm():
    """Lazy singleton; downloads checkpoint from Hugging Face on first use."""
    global _tfm_model
    if _tfm is None:
        raise RuntimeError("Install the `timesfm` package and PyTorch to run forecasts.")
    if _tfm_model is not None:
        return _tfm_model

    backend = "gpu" if os.environ.get("TIMESFM_CUDA", "").lower() in ("1", "true", "yes") else "cpu"
    horizon_len = int(os.environ.get("TIMESFM_HORIZON_LEN", "32"))

    _tfm_model = _tfm.TimesFm(
        hparams=_tfm.TimesFmHparams(
            backend=backend,
            per_core_batch_size=1,
            horizon_len=horizon_len,
            context_len=512,
            point_forecast_mode="mean",
        ),
        checkpoint=_tfm.TimesFmCheckpoint(
            version="torch",
            huggingface_repo_id=_repo_id(),
        ),
    )
    return _tfm_model


def forecast_gap_decimals(
    gap_decimal: np.ndarray,
    horizon: int,
    max_context: int,
) -> GapForecastResult:
    """
    gap_decimal: historical overnight gaps as fractions (e.g. -0.02 .. 0.15).
    """
    model = get_timesfm()
    g = np.asarray(gap_decimal, dtype=np.float32).ravel()
    if g.size < 32:
        raise ValueError("Need at least ~32 valid overnight gap observations.")

    ctx = min(max_context, len(g))
    mean_fc, full_fc = model.forecast(
        inputs=[g],
        freq=[0],
        forecast_context_len=ctx,
        normalize=True,
    )
    # full_fc: (batch, horizon_len, 1 + n_quantiles)
    mean_fc = np.asarray(mean_fc[0], dtype=float)
    full_fc = np.asarray(full_fc[0], dtype=float)
    h = min(horizon, mean_fc.shape[0])
    mean_fc = mean_fc[:h]
    q_block = full_fc[:h, 1:] if full_fc.ndim == 2 else full_fc[:h, 1:]
    return GapForecastResult(
        point=mean_fc,
        quantiles=q_block,
        context_used=ctx,
        horizon=h,
    )


def risk_score_pct(
    result: GapForecastResult,
    threshold_pct: float,
) -> Tuple[float, float, float]:
    """
    Returns (score_0_100, max_point_gap_pct, max_high_quantile_gap_pct).
    Quantiles are experimental (uncalibrated per TimesFM docs).
    """
    pt = result.point * 100.0
    max_pt = float(np.max(pt)) if pt.size else 0.0

    max_q_hi = max_pt
    if result.quantiles.size and result.quantiles.ndim == 2:
        hi = result.quantiles[:, -1]
        max_q_hi = float(np.max(hi)) * 100.0

    tail = max(max_pt, max_q_hi)
    score = 100.0 / (1.0 + np.exp(-0.25 * (tail - threshold_pct)))
    return float(score), max_pt, max_q_hi


def risk_score_down_pct(
    result: GapForecastResult,
    threshold_pct: float,
) -> Tuple[float, float, float]:
    """
    Downside analogue of risk_score_pct: emphasize large negative overnight gaps
    (prior close → next open) vs threshold. Uses min point forecast and lower quantile band.
    Returns (score_0_100, min_point_gap_pct, min_low_quantile_gap_pct).
    """
    pt = result.point * 100.0
    min_pt = float(np.min(pt)) if pt.size else 0.0

    min_q_lo = min_pt
    if result.quantiles.size and result.quantiles.ndim == 2:
        lo = result.quantiles[:, 0]
        min_q_lo = float(np.min(lo)) * 100.0

    tail = min(min_pt, min_q_lo)
    down_mag = -tail if tail < 0 else 0.0
    score = 100.0 / (1.0 + np.exp(-0.25 * (down_mag - threshold_pct)))
    return float(score), min_pt, min_q_lo
