from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import signal

from .config import CONFIG

# Type alias for a 1-D float array.
FloatArray = npt.NDArray[np.floating]

def _detrend(x: FloatArray) -> FloatArray:
    """Remove linear trend and DC offset."""
    if len(x) < 3:
        return x - np.mean(x)
    return signal.detrend(x, type="linear")


def _reversal_rate(s: FloatArray, fs: int) -> float:
    """Steering direction reversals (exceeding dead-band) per minute."""
    cfg = CONFIG.features
    if len(s) < 3:
        return 0.0
    v: FloatArray = np.gradient(s)
    sv: FloatArray = np.sign(v)
    tp_idx: list[int] = [0]
    for i in range(1, len(sv)):
        if sv[i] != 0 and sv[i - 1] != 0 and sv[i] != sv[i - 1]:
            tp_idx.append(i)
    tp_idx.append(len(s) - 1)
    tp: FloatArray = s[tp_idx]
    count: int = 0
    last: float = float(tp[0])
    for val in tp[1:]:
        if abs(float(val) - last) >= cfg.reversal_gap:
            count += 1
            last = float(val)
    minutes: float = len(s) / fs / 60.0
    return count / minutes if minutes > 0 else 0.0


def _hold_fraction(s: FloatArray) -> float:
    """Fraction of window where steering velocity ≈ 0."""
    cfg = CONFIG.features
    if len(s) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(s)) < cfg.hold_vel_thresh))


def _lf_hf_ratio(s: FloatArray, fs: int) -> float:
    """Low-frequency / high-frequency steering power ratio (Welch PSD)."""
    cfg = CONFIG.features
    if len(s) < 16:
        return 0.0
    nperseg: int = min(256, len(s))
    f: FloatArray
    pxx: FloatArray
    f, pxx = signal.welch(s, fs=fs, nperseg=nperseg)

    lf_mask: FloatArray = (f >= cfg.lf_band[0]) & (f < cfg.lf_band[1])
    hf_mask: FloatArray = (f >= cfg.hf_band[0]) & (f < cfg.hf_band[1])
    lf_p: float = float(np.trapezoid(pxx[lf_mask], f[lf_mask]))
    hf_p: float = float(np.trapezoid(pxx[hf_mask], f[hf_mask]))
    return lf_p / hf_p if hf_p > 1e-12 else 0.0


def _steering_prediction_errors(s: FloatArray) -> FloatArray:
    """Second-order Taylor prediction errors (for steering entropy)."""
    if len(s) < 4:
        return np.array([], dtype=float)
    pred: FloatArray = 3 * s[2:-1] - 3 * s[1:-2] + s[0:-3]
    return s[3:] - pred


def _steering_entropy(errors: FloatArray, alpha: float) -> float:
    """Nakayama/Boyle steering entropy from prediction errors."""
    if len(errors) == 0 or not np.isfinite(alpha) or alpha <= 0:
        return 0.0
    k5, k25, k1, k05 = CONFIG.features.entropy_bins_k
    edges: FloatArray = np.array([
        -np.inf, -k5 * alpha, -k25 * alpha, -k1 * alpha, -k05 * alpha,
        k05 * alpha, k1 * alpha, k25 * alpha, k5 * alpha, np.inf,
    ])
    counts: FloatArray
    counts, _ = np.histogram(errors, bins=edges)
    p: FloatArray = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(9))

def extract_features(
    steering: FloatArray | list[float],
    fs: int,
    alpha: float,
) -> dict[str, float]:
    """Compute all 11 steering-only features from a steering array"""
    cfg = CONFIG.features
    s: FloatArray = np.asarray(steering, dtype=float)
    sd: FloatArray = _detrend(s)
    dsd: FloatArray = np.diff(sd)
    ds_raw: FloatArray = np.diff(s)
    abs_ds: FloatArray = np.abs(ds_raw)

    # Kurtosis (excess, Fisher definition)
    kurtosis: float = 0.0
    if len(dsd) > 3 and dsd.std() > 1e-9:
        kurtosis = float(
            ((dsd - dsd.mean()) ** 4).mean() / (dsd.std() ** 4) - 3.0
        )

    # Large-correction rate
    lcr: float = 0.0
    if len(abs_ds) > 0:
        minutes: float = len(s) / fs / 60.0
        lcr = float((abs_ds > cfg.large_corr_thresh).sum() / minutes)

    return {
        "steer_sd": float(np.std(s)),
        "steer_range": float(np.ptp(s)),
        "steer_vel_sd": float(np.std(dsd)),
        "steer_vel_mean_abs": float(np.mean(np.abs(dsd))),
        "reversal_rate": _reversal_rate(sd, fs),
        "hold_fraction": _hold_fraction(sd),
        "lf_hf_ratio": _lf_hf_ratio(sd, fs),
        "steering_entropy": _steering_entropy(
            _steering_prediction_errors(sd), alpha),
        "max_abs_steer_rate": float(abs_ds.max()) if len(abs_ds) else 0.0,
        "steer_rate_kurtosis": kurtosis,
        "large_correction_rate": lcr,
    }
