"""Energy-supported and dynamic range-bin tracking kernel."""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal
from .preprocessing import SignalRuntime, array_key, batch_filter, batch_welch

RADAR_FS_DEFAULT = 120.0
WINDOW_SECONDS = 5.0
HR_LOW_BPM = 45.0
HR_HIGH_BPM = 150.0
LOW_HZ = HR_LOW_BPM / 60.0
HIGH_HZ = HR_HIGH_BPM / 60.0
MIN_RANGE_M = 0.30
MAX_RANGE_M = 2.00
ENERGY_THRESHOLD = 0.50
ENERGY_THRESHOLD_FALLBACK = 0.35
MIN_ENERGY_BINS = 5
MAX_ENERGY_BINS = 24
TOP_ENERGY_BINS_FALLBACK = 12
TRACK_NUM_SEGMENTS = 5
TRACK_JUMP_PENALTY = 0.20
TRACK_MAX_BINS = 18
PSD_NFFT_FULL = 4096
PSD_NFFT_SEGMENT = 1024
PENALIZE_LOW_PEAK_FOR_TRACKING = True
LOW_PEAK_PENALTY_BPM = 65.0

def safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default

def finite_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_iqr": np.nan,
        }
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_iqr": float(q75 - q25),
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.sum(values[mask] * weights[mask]) / (np.sum(weights[mask]) + 1e-12))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cutoff = 0.5 * np.sum(weights)
    return float(values[np.searchsorted(cdf, cutoff)])


def entropy_from_weights(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    total = float(np.sum(w))
    if total <= 1e-12:
        return np.nan
    p = w / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


# ============================================================
# 3. MAT loading

def phase_signal(matrix_10s: np.ndarray, bin_idx: int) -> np.ndarray:
    iq = matrix_10s[int(bin_idx), :]
    phase = np.unwrap(np.angle(iq))
    phase = np.asarray(phase, dtype=float)
    phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        phase = signal.detrend(phase)
    except Exception:
        phase = phase - np.mean(phase)
    return phase


def bandpass(x: np.ndarray, fs: float, low_hz: float = LOW_HZ, high_hz: float = HIGH_HZ) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 20:
        return x
    nyq = 0.5 * fs
    high_hz = min(high_hz, nyq * 0.95)
    if high_hz <= low_hz:
        return x - np.mean(x)
    sos = signal.butter(4, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
    try:
        y = signal.sosfiltfilt(sos, x)
    except Exception:
        y = signal.sosfilt(sos, x)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad > 1e-12:
        return (x - med) / (1.4826 * mad)
    std = np.std(x)
    if std > 1e-12:
        return (x - np.mean(x)) / std
    return x - np.mean(x)


def compute_psd(x: np.ndarray, fs: float, nfft: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if len(x) < 20:
        return np.array([]), np.array([])
    nperseg = min(len(x), int(round(WINDOW_SECONDS * fs)))
    nperseg = max(64, nperseg)
    nperseg = min(nperseg, len(x))
    noverlap = 0 if nperseg == len(x) else nperseg // 2
    freqs, psd = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=max(nfft, nperseg),
        detrend="constant",
        scaling="density",
    )
    psd = np.asarray(psd, dtype=float)
    psd[~np.isfinite(psd)] = 0.0
    return freqs, psd


def spectral_summary_for_bin(matrix_10s: np.ndarray, bin_idx: int, fs: float, nfft: int = PSD_NFFT_FULL) -> Dict[str, float]:
    out = {
        "peak_hr": np.nan, "peak_hz": np.nan, "peak_power": np.nan,
        "band_power": np.nan, "peak_ratio": np.nan, "snr_like": np.nan,
        "centroid_hr": np.nan, "bandwidth_hz": np.nan,
        "top1_top2_ratio": np.nan, "top1_top2_hr_diff": np.nan,
        "phase_std": np.nan, "phase_ptp": np.nan,
        "filtered_std": np.nan, "filtered_ptp": np.nan,
        "amp_mean": np.nan, "amp_std": np.nan, "amp_cv": np.nan,
    }

    bin_idx = int(bin_idx)
    if bin_idx < 0 or bin_idx >= matrix_10s.shape[0]:
        return out

    amp = np.abs(matrix_10s[bin_idx, :])
    ph = phase_signal(matrix_10s, bin_idx)
    filt = robust_zscore(bandpass(ph, fs))

    freqs, psd = compute_psd(filt, fs, nfft=nfft)
    if len(freqs) == 0:
        return out

    mask = (freqs >= LOW_HZ) & (freqs <= min(HIGH_HZ, fs * 0.5 * 0.95))
    if not np.any(mask):
        return out

    bf = freqs[mask]
    bp = psd[mask]
    total = float(np.sum(bp) + 1e-12)

    peaks, _ = signal.find_peaks(bp)
    if len(peaks) > 0:
        order = peaks[np.argsort(bp[peaks])[::-1]]
    else:
        order = np.argsort(bp)[::-1]

    if len(order) == 0:
        return out

    i1 = int(order[0])
    f1 = float(bf[i1])
    p1 = float(bp[i1])

    centroid = float(np.sum(bf * bp) / total)
    bandwidth = float(np.sqrt(np.sum(((bf - centroid) ** 2) * bp) / total))

    noise_mask = np.abs(bf - f1) > 0.10
    if np.any(noise_mask):
        noise = float(np.mean(bp[noise_mask]) + 1e-12)
    else:
        noise = float(np.mean(bp) + 1e-12)

    out.update({
        "peak_hr": f1 * 60.0,
        "peak_hz": f1,
        "peak_power": p1,
        "band_power": total,
        "peak_ratio": float(p1 / total),
        "snr_like": float(p1 / noise),
        "centroid_hr": centroid * 60.0,
        "bandwidth_hz": bandwidth,
        "phase_std": float(np.std(ph)),
        "phase_ptp": float(np.ptp(ph)),
        "filtered_std": float(np.std(filt)),
        "filtered_ptp": float(np.ptp(filt)),
        "amp_mean": float(np.mean(amp)),
        "amp_std": float(np.std(amp)),
        "amp_cv": float(np.std(amp) / (np.mean(amp) + 1e-12)),
    })

    if len(order) >= 2:
        i2 = int(order[1])
        f2 = float(bf[i2])
        p2 = float(bp[i2])
        out["top1_top2_ratio"] = float(p1 / (p2 + 1e-12))
        out["top1_top2_hr_diff"] = float(abs(f1 - f2) * 60.0)

    return out


# ============================================================
# 5. Energy-mask bin selection
# ============================================================

def valid_range_bins(matrix_10s: np.ndarray, range_grid: Optional[np.ndarray]) -> np.ndarray:
    n_bins = matrix_10s.shape[0]
    if range_grid is not None and len(range_grid) == n_bins:
        mask = (range_grid >= MIN_RANGE_M) & (range_grid <= MAX_RANGE_M)
        idx = np.where(mask)[0]
        if len(idx) > 0:
            return idx.astype(int)
    return np.arange(n_bins, dtype=int)


def select_energy_bins(matrix_10s: np.ndarray, range_grid: Optional[np.ndarray]) -> Dict:
    n_bins = matrix_10s.shape[0]
    valid_bins = valid_range_bins(matrix_10s, range_grid)

    energy_all = np.sum(np.abs(matrix_10s), axis=1)
    energy_all = np.asarray(energy_all, dtype=float)
    energy_all[~np.isfinite(energy_all)] = 0.0

    valid_energy = energy_all[valid_bins]
    max_energy = float(np.max(valid_energy)) if len(valid_energy) else 0.0

    if max_energy <= 1e-12:
        fallback = valid_bins[:min(len(valid_bins), TOP_ENERGY_BINS_FALLBACK)]
        energy_norm_all = np.zeros(n_bins, dtype=float)
        return {
            "selected_bins": fallback.astype(int),
            "valid_bins": valid_bins.astype(int),
            "energy_all": energy_all,
            "energy_norm_all": energy_norm_all,
            "threshold_used": np.nan,
            "selection_mode": "fallback_zero_energy",
        }

    energy_norm_all = energy_all / (max_energy + 1e-12)
    valid_energy_norm = energy_norm_all[valid_bins]

    selected = valid_bins[valid_energy_norm >= ENERGY_THRESHOLD]
    threshold_used = ENERGY_THRESHOLD
    selection_mode = "threshold_0.50"

    # Relax the energy mask only when the primary threshold yields too few
    # tracking candidates.
    if len(selected) < MIN_ENERGY_BINS:
        selected = valid_bins[valid_energy_norm >= ENERGY_THRESHOLD_FALLBACK]
        threshold_used = ENERGY_THRESHOLD_FALLBACK
        selection_mode = "threshold_0.35"

    if len(selected) < MIN_ENERGY_BINS:
        order = valid_bins[np.argsort(energy_all[valid_bins])[::-1]]
        selected = order[:min(TOP_ENERGY_BINS_FALLBACK, len(order))]
        threshold_used = np.nan
        selection_mode = "top_energy_fallback"

    if len(selected) > MAX_ENERGY_BINS:
        selected = selected[np.argsort(energy_all[selected])[::-1][:MAX_ENERGY_BINS]]
        selected = np.sort(selected)
        selection_mode += "_top_cap"

    selected = np.asarray(sorted(set(map(int, selected))), dtype=int)

    return {
        "selected_bins": selected,
        "valid_bins": valid_bins.astype(int),
        "energy_all": energy_all,
        "energy_norm_all": energy_norm_all,
        "threshold_used": float(threshold_used) if np.isfinite(threshold_used) else np.nan,
        "selection_mode": selection_mode,
    }


def energy_distribution_features(sel_info: Dict, original_row: pd.Series) -> Dict[str, float | str]:
    selected = np.asarray(sel_info["selected_bins"], dtype=int)
    valid_bins = np.asarray(sel_info["valid_bins"], dtype=int)
    energy_all = np.asarray(sel_info["energy_all"], dtype=float)
    energy_norm = np.asarray(sel_info["energy_norm_all"], dtype=float)

    out: Dict[str, float | str] = {
        "erg_selection_mode": str(sel_info["selection_mode"]),
        "erg_energy_threshold_used": sel_info["threshold_used"],
        "erg_num_selected_bins": int(len(selected)),
        "erg_num_valid_range_bins": int(len(valid_bins)),
        "erg_selected_bins": "|".join(map(str, selected.tolist())),
    }

    if len(selected) == 0:
        for k in [
            "erg_top_energy_bin", "erg_top_energy_norm", "erg_selected_bin_min",
            "erg_selected_bin_max", "erg_selected_bin_span", "erg_selected_bin_center",
            "erg_energy_concentration_top1", "erg_energy_concentration_selected",
            "erg_energy_entropy_selected", "erg_selected_bin_energy_norm_mean",
            "erg_selected_bin_energy_norm_std", "erg_selected_bin_energy_norm_min",
            "erg_selected_bin_energy_norm_max", "erg_selected_bin_energy_norm_median",
            "erg_original_selected_bin_energy_norm", "erg_original_selected_bin_energy_rank",
        ]:
            out[k] = np.nan
        out["erg_original_selected_bin_in_mask"] = 0
        return out

    selected_energy = energy_all[selected]
    selected_norm = energy_norm[selected]

    top_bin = int(selected[np.argmax(selected_energy)])
    valid_order = valid_bins[np.argsort(energy_all[valid_bins])[::-1]]
    rank_map = {int(b): int(i + 1) for i, b in enumerate(valid_order)}

    original_selected_bin = safe_float(original_row.get("selected_bin", np.nan), default=np.nan)
    if np.isfinite(original_selected_bin):
        original_selected_bin = int(round(original_selected_bin))
        if 0 <= original_selected_bin < len(energy_norm):
            orig_energy_norm = float(energy_norm[original_selected_bin])
            orig_in_mask = int(original_selected_bin in set(selected.tolist()))
            orig_rank = rank_map.get(original_selected_bin, np.nan)
        else:
            orig_energy_norm = np.nan
            orig_in_mask = 0
            orig_rank = np.nan
    else:
        orig_energy_norm = np.nan
        orig_in_mask = 0
        orig_rank = np.nan

    selected_total_energy = float(np.sum(selected_energy) + 1e-12)
    valid_total_energy = float(np.sum(energy_all[valid_bins]) + 1e-12)

    out.update({
        "erg_top_energy_bin": int(top_bin),
        "erg_top_energy_norm": float(energy_norm[top_bin]),
        "erg_selected_bin_min": int(np.min(selected)),
        "erg_selected_bin_max": int(np.max(selected)),
        "erg_selected_bin_span": int(np.max(selected) - np.min(selected)),
        "erg_selected_bin_center": float(np.average(selected, weights=selected_energy + 1e-12)),
        "erg_energy_concentration_top1": float(np.max(selected_energy) / valid_total_energy),
        "erg_energy_concentration_selected": float(selected_total_energy / valid_total_energy),
        "erg_energy_entropy_selected": entropy_from_weights(selected_energy),
        "erg_selected_bin_energy_norm_mean": float(np.mean(selected_norm)),
        "erg_selected_bin_energy_norm_std": float(np.std(selected_norm)),
        "erg_selected_bin_energy_norm_min": float(np.min(selected_norm)),
        "erg_selected_bin_energy_norm_max": float(np.max(selected_norm)),
        "erg_selected_bin_energy_norm_median": float(np.median(selected_norm)),
        "erg_original_selected_bin_energy_norm": orig_energy_norm,
        "erg_original_selected_bin_in_mask": int(orig_in_mask),
        "erg_original_selected_bin_energy_rank": orig_rank,
    })

    return out


# ============================================================
# 6. erg_* spectral features over energy-selected bins
# ============================================================

def extract_erg_spectral_features(matrix_10s: np.ndarray, selected_bins: np.ndarray, energy_norm: np.ndarray, fs: float) -> Dict[str, float]:
    out: Dict[str, float] = {}

    if len(selected_bins) == 0:
        for k in [
            "erg_peak_hr_mean", "erg_peak_hr_median", "erg_peak_hr_std", "erg_peak_hr_iqr",
            "erg_peak_hr_weighted_mean", "erg_peak_hr_weighted_median",
            "erg_snr_like_mean", "erg_snr_like_median", "erg_snr_like_std", "erg_snr_like_max",
            "erg_snr_weighted_mean", "erg_peak_ratio_mean", "erg_peak_ratio_median",
            "erg_peak_ratio_std", "erg_peak_ratio_max", "erg_peak_ratio_weighted_mean",
            "erg_agreement_count_5bpm", "erg_agreement_count_8bpm",
            "erg_final_candidate_hr", "erg_final_candidate_snr", "erg_final_candidate_peak_ratio",
            "erg_final_candidate_bin", "erg_final_candidate_energy_norm",
            "erg_final_candidate_x2_if_low", "erg_final_candidate_half_if_high",
        ]:
            out[k] = np.nan
        return out

    rows = []
    for b in selected_bins:
        spec = spectral_summary_for_bin(matrix_10s, int(b), fs=fs, nfft=PSD_NFFT_FULL)
        w_energy = float(energy_norm[int(b)]) if int(b) < len(energy_norm) else np.nan
        quality_weight = max(spec["snr_like"], 0.0) * max(spec["peak_ratio"], 0.0) * max(w_energy, 0.0)
        rows.append({"bin": int(b), "energy_norm": w_energy, "quality_weight": quality_weight, **spec})

    sdf = pd.DataFrame(rows)
    peak_hr = pd.to_numeric(sdf["peak_hr"], errors="coerce").to_numpy(dtype=float)
    snr = pd.to_numeric(sdf["snr_like"], errors="coerce").to_numpy(dtype=float)
    ratio = pd.to_numeric(sdf["peak_ratio"], errors="coerce").to_numpy(dtype=float)
    energy_w = pd.to_numeric(sdf["energy_norm"], errors="coerce").to_numpy(dtype=float)
    q_w = pd.to_numeric(sdf["quality_weight"], errors="coerce").to_numpy(dtype=float)

    out.update(finite_stats(peak_hr, "erg_peak_hr"))
    out["erg_peak_hr_weighted_mean"] = weighted_mean(peak_hr, q_w)
    out["erg_peak_hr_weighted_median"] = weighted_median(peak_hr, q_w)

    out.update(finite_stats(snr, "erg_snr_like"))
    out["erg_snr_weighted_mean"] = weighted_mean(snr, energy_w)

    out.update(finite_stats(ratio, "erg_peak_ratio"))
    out["erg_peak_ratio_weighted_mean"] = weighted_mean(ratio, energy_w)

    candidate_hr = out["erg_peak_hr_weighted_median"]
    if not np.isfinite(candidate_hr):
        candidate_hr = out["erg_peak_hr_median"]

    if np.isfinite(candidate_hr):
        out["erg_agreement_count_5bpm"] = int(np.sum(np.abs(peak_hr - candidate_hr) <= 5.0))
        out["erg_agreement_count_8bpm"] = int(np.sum(np.abs(peak_hr - candidate_hr) <= 8.0))
    else:
        out["erg_agreement_count_5bpm"] = 0
        out["erg_agreement_count_8bpm"] = 0

    valid_q = np.isfinite(q_w)
    if valid_q.any() and np.nanmax(q_w) > 0:
        best_i = int(np.nanargmax(q_w))
    else:
        best_i = int(np.nanargmax(energy_w))

    best = sdf.iloc[best_i]
    out["erg_final_candidate_hr"] = float(best["peak_hr"]) if np.isfinite(best["peak_hr"]) else candidate_hr
    out["erg_final_candidate_snr"] = float(best["snr_like"]) if np.isfinite(best["snr_like"]) else np.nan
    out["erg_final_candidate_peak_ratio"] = float(best["peak_ratio"]) if np.isfinite(best["peak_ratio"]) else np.nan
    out["erg_final_candidate_bin"] = int(best["bin"])
    out["erg_final_candidate_energy_norm"] = float(best["energy_norm"])

    final_hr = out["erg_final_candidate_hr"]
    out["erg_final_candidate_x2_if_low"] = (
        float(final_hr * 2.0)
        if np.isfinite(final_hr) and final_hr < 65.0 and final_hr * 2.0 <= HR_HIGH_BPM
        else final_hr
    )
    out["erg_final_candidate_half_if_high"] = (
        float(final_hr / 2.0)
        if np.isfinite(final_hr) and final_hr > 120.0 and final_hr / 2.0 >= HR_LOW_BPM
        else final_hr
    )

    return out


# ============================================================
# 7. rt_* range-bin tracking over energy-selected bins
# ============================================================

def compute_tracking_score_for_bin_segment(segment_matrix: np.ndarray, bin_idx: int, fs: float) -> Dict[str, float]:
    spec = spectral_summary_for_bin(segment_matrix, int(bin_idx), fs=fs, nfft=PSD_NFFT_SEGMENT)
    peak_hr = spec["peak_hr"]
    snr = spec["snr_like"]
    ratio = spec["peak_ratio"]
    amp_mean = spec["amp_mean"]

    if not np.isfinite(peak_hr):
        score = -np.inf
    else:
        score = math.log1p(max(snr, 0.0)) * max(ratio, 0.0) * math.log1p(max(amp_mean, 0.0))
        if PENALIZE_LOW_PEAK_FOR_TRACKING and peak_hr < LOW_PEAK_PENALTY_BPM:
            score *= 0.65
        if not np.isfinite(score):
            score = -np.inf

    return {"score": float(score), "peak_hr": spec["peak_hr"], "snr_like": spec["snr_like"], "peak_ratio": spec["peak_ratio"]}


def build_tracking_quality_table(matrix_10s: np.ndarray, candidate_bins: np.ndarray, fs: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    n_time = matrix_10s.shape[1]
    edges = np.linspace(0, n_time, TRACK_NUM_SEGMENTS + 1).round().astype(int)

    n_seg = TRACK_NUM_SEGMENTS
    n_bins = len(candidate_bins)

    scores = np.full((n_seg, n_bins), -np.inf, dtype=float)
    peak_hr = np.full((n_seg, n_bins), np.nan, dtype=float)
    snr_like = np.full((n_seg, n_bins), np.nan, dtype=float)
    peak_ratio = np.full((n_seg, n_bins), np.nan, dtype=float)

    for t in range(n_seg):
        start = int(edges[t])
        end = int(edges[t + 1])
        if end - start < 20:
            continue
        seg = matrix_10s[:, start:end]
        for j, b in enumerate(candidate_bins):
            try:
                q = compute_tracking_score_for_bin_segment(seg, int(b), fs=fs)
            except Exception:
                continue
            scores[t, j] = q["score"]
            peak_hr[t, j] = q["peak_hr"]
            snr_like[t, j] = q["snr_like"]
            peak_ratio[t, j] = q["peak_ratio"]

    return scores, {"peak_hr": peak_hr, "snr_like": snr_like, "peak_ratio": peak_ratio}


def dp_track_bins(scores: np.ndarray, candidate_bins: np.ndarray) -> List[int]:
    n_seg, n_bins = scores.shape
    if n_bins == 0:
        return []

    clean = np.where(np.isfinite(scores), scores, -1e9)
    bins_float = candidate_bins.astype(float)

    dp = np.full((n_seg, n_bins), -np.inf, dtype=float)
    prev = np.full((n_seg, n_bins), -1, dtype=int)
    dp[0, :] = clean[0, :]

    # Maximize cumulative spectral quality while penalizing abrupt
    # range-bin transitions.
    for t in range(1, n_seg):
        for j in range(n_bins):
            jump_cost = TRACK_JUMP_PENALTY * np.abs(bins_float[j] - bins_float)
            vals = dp[t - 1, :] - jump_cost + clean[t, j]
            k = int(np.argmax(vals))
            dp[t, j] = vals[k]
            prev[t, j] = k

    last = int(np.argmax(dp[-1, :]))
    idx_path = [last]
    for t in range(n_seg - 1, 0, -1):
        last = int(prev[t, last])
        if last < 0:
            last = idx_path[-1]
        idx_path.append(last)

    idx_path = idx_path[::-1]
    return [int(candidate_bins[i]) for i in idx_path]


def extract_rt_features_from_energy_bins(matrix_10s: np.ndarray, selected_bins: np.ndarray, energy_norm: np.ndarray, fs: float, original_row: pd.Series) -> Dict[str, float | str]:
    if len(selected_bins) == 0:
        return {
            "rt_candidate_source": "energy_mask_empty",
            "rt_tracking_bins": "",
            "rt_path_bins": "",
            "rt_tracked_final_bin": np.nan,
            "rt_stability_score": np.nan,
            "rt_jump_count": np.nan,
            "rt_final_peak_hr": np.nan,
        }

    if len(selected_bins) > TRACK_MAX_BINS:
        selected_bins = np.asarray(selected_bins, dtype=int)
        top = selected_bins[np.argsort(energy_norm[selected_bins])[::-1][:TRACK_MAX_BINS]]
        candidate_bins = np.asarray(sorted(top.tolist()), dtype=int)
        source = "energy_mask_top_energy_for_tracking"
    else:
        candidate_bins = np.asarray(sorted(selected_bins.tolist()), dtype=int)
        source = "energy_mask_all_selected_for_tracking"

    scores, aux = build_tracking_quality_table(matrix_10s, candidate_bins, fs=fs)
    path_bins = dp_track_bins(scores, candidate_bins)

    if len(path_bins) == 0:
        return {
            "rt_candidate_source": source,
            "rt_tracking_bins": "|".join(map(str, candidate_bins.tolist())),
            "rt_path_bins": "",
            "rt_tracked_final_bin": np.nan,
            "rt_stability_score": np.nan,
            "rt_jump_count": np.nan,
            "rt_final_peak_hr": np.nan,
        }

    path_bins_arr = np.asarray(path_bins, dtype=float)
    path_idx = [int(np.where(candidate_bins == b)[0][0]) for b in path_bins]

    path_scores = np.asarray([scores[t, path_idx[t]] for t in range(len(path_idx))], dtype=float)
    path_hrs = np.asarray([aux["peak_hr"][t, path_idx[t]] for t in range(len(path_idx))], dtype=float)
    path_snrs = np.asarray([aux["snr_like"][t, path_idx[t]] for t in range(len(path_idx))], dtype=float)
    path_ratios = np.asarray([aux["peak_ratio"][t, path_idx[t]] for t in range(len(path_idx))], dtype=float)

    tracked_final_bin = int(round(float(np.nanmedian(path_bins_arr))))
    tracked_final_bin = int(max(0, min(tracked_final_bin, matrix_10s.shape[0] - 1)))
    final_spec = spectral_summary_for_bin(matrix_10s, tracked_final_bin, fs=fs, nfft=PSD_NFFT_FULL)

    jumps = np.abs(np.diff(path_bins_arr)) if len(path_bins_arr) >= 2 else np.array([], dtype=float)
    if len(path_bins_arr) > 0:
        _, counts = np.unique(path_bins_arr, return_counts=True)
        stability = float(np.max(counts) / len(path_bins_arr))
    else:
        stability = np.nan

    original_selected_bin = safe_float(original_row.get("selected_bin", np.nan), default=np.nan)
    if np.isfinite(original_selected_bin):
        original_selected_bin = int(round(original_selected_bin))
        anchor_match = float(np.mean(path_bins_arr == original_selected_bin)) if len(path_bins_arr) else np.nan
        final_vs_anchor = int(tracked_final_bin - original_selected_bin)
    else:
        anchor_match = np.nan
        final_vs_anchor = np.nan

    selected_specs = []
    for b in candidate_bins:
        spec = spectral_summary_for_bin(matrix_10s, int(b), fs=fs, nfft=PSD_NFFT_FULL)
        selected_specs.append((int(b), spec))

    selected_hrs = np.asarray([s["peak_hr"] for _, s in selected_specs], dtype=float)
    tracked_hr = final_spec["peak_hr"]

    if np.isfinite(tracked_hr):
        agree5 = int(np.sum(np.abs(selected_hrs - tracked_hr) <= 5.0))
        agree8 = int(np.sum(np.abs(selected_hrs - tracked_hr) <= 8.0))
    else:
        agree5 = 0
        agree8 = 0

    return {
        "rt_candidate_source": source,
        "rt_tracking_bins": "|".join(map(str, candidate_bins.tolist())),
        "rt_path_bins": "|".join(map(str, path_bins)),
        "rt_tracked_final_bin": int(tracked_final_bin),

        "rt_path_bin_mean": float(np.nanmean(path_bins_arr)),
        "rt_path_bin_median": float(np.nanmedian(path_bins_arr)),
        "rt_path_bin_std": float(np.nanstd(path_bins_arr)),
        "rt_path_bin_min": float(np.nanmin(path_bins_arr)),
        "rt_path_bin_max": float(np.nanmax(path_bins_arr)),
        "rt_path_bin_range": float(np.nanmax(path_bins_arr) - np.nanmin(path_bins_arr)),

        "rt_jump_count": int(np.sum(jumps > 0)) if len(jumps) else 0,
        "rt_mean_abs_jump": float(np.nanmean(jumps)) if len(jumps) else 0.0,
        "rt_max_abs_jump": float(np.nanmax(jumps)) if len(jumps) else 0.0,
        "rt_stability_score": stability,
        "rt_anchor_match_fraction": anchor_match,
        "rt_final_bin_minus_original_selected_bin": final_vs_anchor,

        "rt_path_quality_mean": float(np.nanmean(path_scores)),
        "rt_path_quality_std": float(np.nanstd(path_scores)),
        "rt_path_quality_min": float(np.nanmin(path_scores)),
        "rt_path_quality_max": float(np.nanmax(path_scores)),

        "rt_path_peak_hr_mean": float(np.nanmean(path_hrs)),
        "rt_path_peak_hr_median": float(np.nanmedian(path_hrs)),
        "rt_path_peak_hr_std": float(np.nanstd(path_hrs)),
        "rt_path_snr_mean": float(np.nanmean(path_snrs)),
        "rt_path_snr_max": float(np.nanmax(path_snrs)),
        "rt_path_peak_ratio_mean": float(np.nanmean(path_ratios)),
        "rt_path_peak_ratio_max": float(np.nanmax(path_ratios)),

        "rt_final_peak_hr": final_spec["peak_hr"],
        "rt_final_peak_hz": final_spec["peak_hz"],
        "rt_final_snr_like": final_spec["snr_like"],
        "rt_final_peak_ratio": final_spec["peak_ratio"],
        "rt_final_band_power": final_spec["band_power"],
        "rt_final_centroid_hr": final_spec["centroid_hr"],
        "rt_final_bandwidth_hz": final_spec["bandwidth_hz"],
        "rt_final_top1_top2_ratio": final_spec["top1_top2_ratio"],
        "rt_final_top1_top2_hr_diff": final_spec["top1_top2_hr_diff"],
        "rt_final_amp_mean": final_spec["amp_mean"],
        "rt_final_amp_cv": final_spec["amp_cv"],

        "rt_energy_bins_agreement_count_5bpm": agree5,
        "rt_energy_bins_agreement_count_8bpm": agree8,
        "rt_energy_bins_peak_hr_std": float(np.nanstd(selected_hrs)),
        "rt_energy_bins_peak_hr_median": float(np.nanmedian(selected_hrs)),
    }


# ============================================================
# 8. Per-row processing


_REFERENCE_PHASE = phase_signal
_REFERENCE_ROBUST_ZSCORE = robust_zscore


class EnergyTrackingExtractor:
    """Energy features plus axis-batched short-segment Tracking spectra."""

    def __init__(self, runtime: SignalRuntime):
        self.runtime=runtime;self.last_tracking_path=None;self._install()

    def phase(self,matrix:np.ndarray,bin_idx:int)->np.ndarray:
        key=(array_key(matrix),int(bin_idx),"unwrap-angle","linear-detrend")
        return self.runtime.cache.get_or_compute("common_phase",key,lambda:_REFERENCE_PHASE(matrix,bin_idx),(matrix,))

    def filtered(self,x:np.ndarray,fs:float,low_hz:float=LOW_HZ,high_hz:float=HIGH_HZ)->np.ndarray:
        key=(array_key(x),float(fs),float(low_hz),float(high_hz),4,"sosfiltfilt")
        return self.runtime.cache.get_or_compute("common_bandpass",key,lambda:batch_filter(np.asarray(x,float)[None,:],self.runtime.butter_sos(4,fs,low_hz,min(high_hz,.5*fs*.95)))[0],(x,))

    def normalized(self,x:np.ndarray)->np.ndarray:
        return self.runtime.cache.get_or_compute("energy_zscore",(array_key(x),),lambda:_REFERENCE_ROBUST_ZSCORE(x),(x,))

    @staticmethod
    def _summaries(matrix:np.ndarray,bins:np.ndarray,phase:np.ndarray,normalized:np.ndarray,frequency:np.ndarray,psd:np.ndarray,fs:float):
        mask=(frequency>=LOW_HZ)&(frequency<=min(HIGH_HZ,fs*.5*.95));bf=frequency[mask];output=[]
        for row,bin_idx in enumerate(bins):
            bp=psd[row,mask];total=float(np.sum(bp)+1e-12);peaks,_=signal.find_peaks(bp);order=peaks[np.argsort(bp[peaks])[::-1]] if len(peaks) else np.argsort(bp)[::-1];i1=int(order[0]);f1=float(bf[i1]);p1=float(bp[i1]);noise_mask=np.abs(bf-f1)>.10;noise=float(np.mean(bp[noise_mask] if np.any(noise_mask) else bp)+1e-12);amp=np.abs(matrix[int(bin_idx)])
            output.append({"peak_hr":f1*60,"snr_like":float(p1/noise),"peak_ratio":float(p1/total),"amp_mean":float(np.mean(amp))})
        return output

    def tracking_table(self,matrix_10s:np.ndarray,candidate_bins:np.ndarray,fs:float):
        bins=np.asarray(candidate_bins,int);edges=np.linspace(0,matrix_10s.shape[1],TRACK_NUM_SEGMENTS+1).round().astype(int);scores=np.full((TRACK_NUM_SEGMENTS,len(bins)),-np.inf);peak=np.full_like(scores,np.nan);snr=np.full_like(scores,np.nan);ratio=np.full_like(scores,np.nan);sos=self.runtime.butter_sos(4,fs,LOW_HZ,min(HIGH_HZ,.5*fs*.95))
        for segment_index in range(TRACK_NUM_SEGMENTS):
            start,end=int(edges[segment_index]),int(edges[segment_index+1]);segment=matrix_10s[:,start:end]
            if end-start<20 or len(bins)==0:continue
            values=np.nan_to_num(np.asarray(np.unwrap(np.angle(segment[bins]),axis=-1),float),nan=0.0,posinf=0.0,neginf=0.0);phase=signal.detrend(values,axis=-1);filtered=batch_filter(phase,sos);normalized=np.vstack([_REFERENCE_ROBUST_ZSCORE(row) for row in filtered]);nperseg=min(max(64,min(normalized.shape[1],int(round(WINDOW_SECONDS*fs)))),normalized.shape[1]);noverlap=0 if nperseg==normalized.shape[1] else nperseg//2;frequency,psd=batch_welch(normalized,fs,nperseg,noverlap,max(PSD_NFFT_SEGMENT,nperseg));summaries=self._summaries(segment,bins,phase,normalized,frequency,psd,fs)
            for column,item in enumerate(summaries):
                value=math.log1p(max(item["snr_like"],0))*max(item["peak_ratio"],0)*math.log1p(max(item["amp_mean"],0));value*=.65 if PENALIZE_LOW_PEAK_FOR_TRACKING and item["peak_hr"]<LOW_PEAK_PENALTY_BPM else 1.0;scores[segment_index,column]=value;peak[segment_index,column]=item["peak_hr"];snr[segment_index,column]=item["snr_like"];ratio[segment_index,column]=item["peak_ratio"]
        self.last_tracking_path={"scores":scores,"candidate_bins":bins}
        return scores,{"peak_hr":peak,"snr_like":snr,"peak_ratio":ratio}

    def _install(self):
        global phase_signal,bandpass,robust_zscore,build_tracking_quality_table
        phase_signal=self.phase;bandpass=self.filtered;robust_zscore=self.normalized;build_tracking_quality_table=self.tracking_table

