from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat

from ..data.windowing import RADAR_FS as PROTOCOL_RADAR_FS, LOCAL_WINDOW_SECONDS as WINDOW_SEC

WINDOW_IDS = tuple(range(6))
def samples_per_window(fs: float) -> int:
    return int(round(WINDOW_SEC * fs))


# ============================================================
# 0. CONFIG
# ============================================================

# train radar data root directory, contains 1_1, 1_2, 1_3 ...
# Assigned by the public SpatialExtractor from config.yaml.
DATA_ROOT = Path(".")

# Radar sampling rate
RADAR_FS = PROTOCOL_RADAR_FS

# 5s parent window；The number of sampling points is calculated based on a fixed radar sampling rate.
RADAR_POINTS_PER_WINDOW = samples_per_window(RADAR_FS)

# Heart rate search range: 45–150 BPM
LOW_HZ = 45 / 60
HIGH_HZ = 150 / 60

# range bin selection range
MIN_RANGE_M = 0.3
MAX_RANGE_M = 2.0

# After selecting bin, continue to extract features from nearby bins
NEIGHBOR_BINS = 2

# top-k spectrum peak
TOP_K = 5

# 10 second signal using nfft=4096 to make PSD frequency grid finer
PSD_NFFT = 4096

# use full-length Welch or not
USE_FULL_LENGTH_WELCH = True

# the bin selection slightly penalize low-frequency peaks or not
PENALIZE_LOW_PEAK_FOR_BIN_SELECTION = True
LOW_PEAK_PENALTY_HR = 65.0


# ============================================================
# 1. Basic utilities
# ============================================================

def get_mat_field(obj, field_name: str):
    if hasattr(obj, field_name):
        return getattr(obj, field_name)

    if isinstance(obj, dict):
        return obj.get(field_name)

    if isinstance(obj, np.ndarray) and obj.dtype.names and field_name in obj.dtype.names:
        return np.squeeze(obj)[field_name]

    return None


def find_mat_file(sample_dir: Path) -> Path | None:
    candidates = sorted(sample_dir.glob("*complex_range_matrix.mat"))
    if candidates:
        return candidates[0]

    candidates = sorted(sample_dir.glob("*.mat"))
    if candidates:
        return candidates[0]

    return None


def load_radar_mat(sample_dir: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Read the training set radar mat。

    Common structures of training sets:
        save_data.complex_range_matrix
        save_data.range_grid
    """
    mat_path = find_mat_file(sample_dir)

    if mat_path is None:
        raise FileNotFoundError(f"No .mat radar file found in {sample_dir}")

    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    valid_keys = [k for k in data.keys() if not k.startswith("__")]

    matrix = None
    range_grid = None

    if "save_data" in data:
        save_data = data["save_data"]

        matrix = get_mat_field(save_data, "complex_range_matrix")
        range_grid = get_mat_field(save_data, "range_grid")

    if matrix is None and "complex_range_matrix" in data:
        matrix = data["complex_range_matrix"]

    if matrix is None and "radar_data" in data:
        matrix = data["radar_data"]

    if matrix is None and "radar_segment" in data:
        matrix = data["radar_segment"]

    if "range_grid" in data:
        range_grid = data["range_grid"]

    if matrix is None:
        for key in valid_keys:
            value = data[key]

            if isinstance(value, np.ndarray) and value.ndim == 2:
                if np.issubdtype(value.dtype, np.number):
                    matrix = value
                    logging.warning("Auto-selected matrix variable %r from %s", key, mat_path)
                    break

    if matrix is None:
        raise KeyError(
            f"No radar matrix found in {mat_path}. "
            f"Available variables: {valid_keys}"
        )

    matrix = np.asarray(matrix)

    if range_grid is not None:
        range_grid = np.ravel(np.asarray(range_grid, dtype=float))
    else:
        range_grid = None

    if matrix.ndim != 2:
        raise ValueError(f"Radar matrix should be 2D, got shape={matrix.shape}")

    # unified as shape = (range_bins, time_points)
    if range_grid is not None:
        if len(range_grid) == matrix.shape[1] and len(range_grid) != matrix.shape[0]:
            matrix = matrix.T
    else:
        if matrix.shape[0] > matrix.shape[1] and matrix.shape[0] >= 1000:
            matrix = matrix.T

    return matrix, range_grid


# ============================================================
# 2. Radar window slicing and preprocessing
# ============================================================

def crop_radar_window(
    matrix: np.ndarray,
    window_id: int,
) -> np.ndarray:
    """
    按 zero-based window_id 切 5 秒 radar。

        window_id = 0 -> 0:600
        ...
        window_id = 5 -> 3000:3600
    """
    if window_id not in WINDOW_IDS:
        raise ValueError(f"window_id must be one of {WINDOW_IDS}, got {window_id}")

    start_idx = window_id * RADAR_POINTS_PER_WINDOW
    end_idx = start_idx + RADAR_POINTS_PER_WINDOW

    n_time = matrix.shape[1]

    if n_time >= end_idx:
        return matrix[:, start_idx:end_idx]

    if n_time > start_idx:
        return matrix[:, start_idx:n_time]

    raise ValueError(
        f"Radar time length is {n_time}, cannot extract window_id={window_id} "
        f"with start index {start_idx}."
    )


def extract_phase_signal(matrix: np.ndarray, bin_index: int) -> np.ndarray:
    iq = matrix[bin_index, :]

    phase = np.unwrap(np.angle(iq))
    phase = np.asarray(phase, dtype=float)
    phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)

    phase = signal.detrend(phase)

    return phase


def bandpass_filter(
    x: np.ndarray,
    fs: float = RADAR_FS,
    low_hz: float = LOW_HZ,
    high_hz: float = HIGH_HZ,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if len(x) < 20:
        return x

    nyquist = 0.5 * fs

    if high_hz >= nyquist:
        high_hz = nyquist * 0.95

    if high_hz <= low_hz:
        return x

    sos = signal.butter(
        N=4,
        Wn=[low_hz / nyquist, high_hz / nyquist],
        btype="bandpass",
        output="sos",
    )

    try:
        y = signal.sosfiltfilt(sos, x)
    except ValueError:
        y = signal.sosfilt(sos, x)

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return y


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    mean = np.mean(x)
    std = np.std(x)

    if std < 1e-8:
        return x - mean

    return (x - mean) / std


# ============================================================
# 3residual. PSD and spectral features
# ============================================================

def compute_welch_psd(
    x: np.ndarray,
    fs: float = RADAR_FS,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if USE_FULL_LENGTH_WELCH:
        nperseg = len(x)
        noverlap = 0
    else:
        nperseg = min(len(x), int(fs * 8))
        nperseg = max(64, nperseg)
        nperseg = min(nperseg, len(x))
        noverlap = nperseg // 2

    freqs, psd = signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=PSD_NFFT,
        detrend="constant",
    )

    return freqs, psd


def top_k_spectral_peaks(
    x: np.ndarray,
    fs: float = RADAR_FS,
    low_hz: float = LOW_HZ,
    high_hz: float = HIGH_HZ,
    top_k: int = TOP_K,
) -> dict[str, float]:
    result: dict[str, float] = {}

    for i in range(1, top_k + 1):
        result[f"top{i}_hr"] = np.nan
        result[f"top{i}_hz"] = np.nan
        result[f"top{i}_power"] = np.nan
        result[f"top{i}_ratio"] = np.nan

    result["peak_hr"] = np.nan
    result["peak_hz"] = np.nan
    result["peak_power"] = np.nan
    result["band_power"] = np.nan
    result["peak_ratio"] = np.nan
    result["centroid_hz"] = np.nan
    result["centroid_hr"] = np.nan
    result["bandwidth_hz"] = np.nan
    result["snr_like"] = np.nan
    result["top1_top2_power_ratio"] = np.nan
    result["top1_top2_hr_diff"] = np.nan

    if len(x) < 20:
        return result

    freqs, psd = compute_welch_psd(x, fs=fs)

    band_mask = (freqs >= low_hz) & (freqs <= high_hz)

    if not np.any(band_mask):
        return result

    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]

    total_power = float(np.sum(band_psd) + 1e-12)

    peaks, _ = signal.find_peaks(band_psd)

    if len(peaks) == 0:
        candidate_indices = np.argsort(band_psd)[::-1]
    else:
        candidate_indices = peaks[np.argsort(band_psd[peaks])[::-1]]

    selected_indices = []
    min_sep_hz = 0.08

    for idx in candidate_indices:
        freq = band_freqs[idx]

        if all(abs(freq - band_freqs[j]) >= min_sep_hz for j in selected_indices):
            selected_indices.append(int(idx))

        if len(selected_indices) >= top_k:
            break

    if len(selected_indices) < top_k:
        all_sorted = np.argsort(band_psd)[::-1]

        for idx in all_sorted:
            if int(idx) not in selected_indices:
                selected_indices.append(int(idx))

            if len(selected_indices) >= top_k:
                break

    for rank, idx in enumerate(selected_indices[:top_k], start=1):
        hz = float(band_freqs[idx])
        power = float(band_psd[idx])
        hr = hz * 60
        ratio = power / total_power

        result[f"top{rank}_hr"] = hr
        result[f"top{rank}_hz"] = hz
        result[f"top{rank}_power"] = power
        result[f"top{rank}_ratio"] = ratio

    result["peak_hr"] = result["top1_hr"]
    result["peak_hz"] = result["top1_hz"]
    result["peak_power"] = result["top1_power"]
    result["peak_ratio"] = result["top1_ratio"]
    result["band_power"] = total_power

    centroid_hz = float(np.sum(band_freqs * band_psd) / total_power)
    centroid_hr = centroid_hz * 60

    bandwidth_hz = float(
        np.sqrt(np.sum(((band_freqs - centroid_hz) ** 2) * band_psd) / total_power)
    )

    result["centroid_hz"] = centroid_hz
    result["centroid_hr"] = centroid_hr
    result["bandwidth_hz"] = bandwidth_hz

    peak_hz = result["peak_hz"]
    peak_power = result["peak_power"]

    exclude_mask = np.abs(band_freqs - peak_hz) > 0.10

    if np.any(exclude_mask):
        noise_power = float(np.mean(band_psd[exclude_mask]) + 1e-12)
    else:
        noise_power = float(np.mean(band_psd) + 1e-12)

    result["snr_like"] = float(peak_power / noise_power)

    if np.isfinite(result["top1_power"]) and np.isfinite(result["top2_power"]):
        result["top1_top2_power_ratio"] = float(
            result["top1_power"] / (result["top2_power"] + 1e-12)
        )

        result["top1_top2_hr_diff"] = float(
            abs(result["top1_hr"] - result["top2_hr"])
        )

    return result


def spectral_features(
    x: np.ndarray,
    fs: float = RADAR_FS,
    prefix: str = "",
) -> dict[str, float]:
    raw = top_k_spectral_peaks(
        x=x,
        fs=fs,
        low_hz=LOW_HZ,
        high_hz=HIGH_HZ,
        top_k=TOP_K,
    )

    return {f"{prefix}{k}": v for k, v in raw.items()}


def time_domain_features(
    raw_phase: np.ndarray,
    filtered_phase: np.ndarray,
    amp: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    features: dict[str, float] = {}

    raw_phase = np.asarray(raw_phase, dtype=float)
    filtered_phase = np.asarray(filtered_phase, dtype=float)
    amp = np.asarray(amp, dtype=float)

    features[f"{prefix}phase_mean"] = float(np.mean(raw_phase))
    features[f"{prefix}phase_std"] = float(np.std(raw_phase))
    features[f"{prefix}phase_ptp"] = float(np.ptp(raw_phase))
    features[f"{prefix}phase_rms"] = float(np.sqrt(np.mean(raw_phase ** 2)))

    features[f"{prefix}filtered_mean"] = float(np.mean(filtered_phase))
    features[f"{prefix}filtered_std"] = float(np.std(filtered_phase))
    features[f"{prefix}filtered_ptp"] = float(np.ptp(filtered_phase))
    features[f"{prefix}filtered_rms"] = float(np.sqrt(np.mean(filtered_phase ** 2)))

    features[f"{prefix}amp_mean"] = float(np.mean(amp))
    features[f"{prefix}amp_std"] = float(np.std(amp))
    features[f"{prefix}amp_max"] = float(np.max(amp))
    features[f"{prefix}amp_min"] = float(np.min(amp))
    features[f"{prefix}amp_cv"] = float(np.std(amp) / (np.mean(amp) + 1e-12))

    return features


# ============================================================
# 4. SNR-based range bin selection
# ============================================================

def get_candidate_bins(
    matrix: np.ndarray,
    range_grid: np.ndarray | None,
) -> list[int]:
    num_bins = matrix.shape[0]

    if range_grid is not None and len(range_grid) == num_bins:
        mask = (range_grid >= MIN_RANGE_M) & (range_grid <= MAX_RANGE_M)
        candidate_bins = np.where(mask)[0].tolist()

        if candidate_bins:
            return candidate_bins

    return list(range(num_bins))


def compute_bin_quality(
    matrix: np.ndarray,
    bin_idx: int,
) -> dict[str, float]:
    amp = np.abs(matrix[bin_idx, :])

    raw_phase = extract_phase_signal(matrix, bin_idx)
    filtered_phase = bandpass_filter(raw_phase, fs=RADAR_FS)
    filtered_phase_z = zscore(filtered_phase)

    spec = top_k_spectral_peaks(
        filtered_phase_z,
        fs=RADAR_FS,
        low_hz=LOW_HZ,
        high_hz=HIGH_HZ,
        top_k=TOP_K,
    )

    amp_mean = float(np.mean(amp))
    amp_std = float(np.std(amp))
    amp_cv = float(amp_std / (amp_mean + 1e-12))

    peak_hr = spec["peak_hr"]
    peak_ratio = spec["peak_ratio"]
    snr_like = spec["snr_like"]
    band_power = spec["band_power"]

    if not np.isfinite(peak_hr):
        score = -np.inf
    else:
        score = float(np.log1p(snr_like) * (peak_ratio + 1e-6))
        score *= float(np.log1p(amp_mean))

        if PENALIZE_LOW_PEAK_FOR_BIN_SELECTION and peak_hr < LOW_PEAK_PENALTY_HR:
            score *= 0.65

        if not np.isfinite(band_power) or band_power <= 1e-12:
            score *= 0.1

    return {
        "bin_idx": float(bin_idx),
        "quality_score": float(score),
        "quality_peak_hr": float(peak_hr),
        "quality_peak_ratio": float(peak_ratio),
        "quality_snr_like": float(snr_like),
        "quality_band_power": float(band_power),
        "quality_amp_mean": amp_mean,
        "quality_amp_cv": amp_cv,
    }


def choose_range_bin_by_snr(
    matrix: np.ndarray,
    range_grid: np.ndarray | None,
) -> tuple[int, float | None, dict[str, float]]:
    candidate_bins = get_candidate_bins(matrix, range_grid)

    qualities = []

    for bin_idx in candidate_bins:
        try:
            q = compute_bin_quality(matrix, bin_idx)
            qualities.append(q)
        except Exception:
            continue

    if not qualities:
        abs_matrix = np.abs(matrix)
        mean_by_bin = np.nanmean(abs_matrix, axis=1)
        selected_bin = int(np.nanargmax(mean_by_bin))

        selected_distance = None

        if range_grid is not None and selected_bin < len(range_grid):
            selected_distance = float(range_grid[selected_bin])

        return selected_bin, selected_distance, {
            "quality_score": np.nan,
            "quality_peak_hr": np.nan,
            "quality_peak_ratio": np.nan,
            "quality_snr_like": np.nan,
            "quality_band_power": np.nan,
            "quality_amp_mean": float(mean_by_bin[selected_bin]),
            "quality_amp_cv": np.nan,
        }

    qualities_sorted = sorted(
        qualities,
        key=lambda item: item["quality_score"],
        reverse=True,
    )

    best = qualities_sorted[0]
    selected_bin = int(best["bin_idx"])

    selected_distance = None

    if range_grid is not None and selected_bin < len(range_grid):
        selected_distance = float(range_grid[selected_bin])

    return selected_bin, selected_distance, best


# ============================================================
# 5. Smart HR candidate
# ============================================================

def smart_hr_candidate(row: dict[str, float | str]) -> float:
    main_peak = row.get("radar_peak_hr_baseline", np.nan)

    if not np.isfinite(main_peak):
        return np.nan

    main_peak = float(main_peak)
    candidate = main_peak

    if main_peak < 65:
        doubled = main_peak * 2

        support_high = False

        neighbor_median = row.get("neighbor_peak_hr_median", np.nan)

        if np.isfinite(neighbor_median):
            if abs(float(neighbor_median) - doubled) <= 8:
                support_high = True

        for i in range(2, TOP_K + 1):
            alt_hr = row.get(f"main_top{i}_hr", np.nan)

            if np.isfinite(alt_hr):
                if abs(float(alt_hr) - doubled) <= 8:
                    support_high = True

        if support_high and 65 <= doubled <= 140:
            candidate = float(doubled)
        else:
            candidate = float(main_peak)

    return candidate


# ============================================================
# 6. Feature extraction for bin/window
# ============================================================

def extract_features_for_bin(
    matrix_10s: np.ndarray,
    bin_idx: int,
    prefix: str,
) -> dict[str, float]:
    amp = np.abs(matrix_10s[bin_idx, :])

    raw_phase = extract_phase_signal(matrix_10s, bin_idx)
    filtered_phase = bandpass_filter(raw_phase, fs=RADAR_FS)
    filtered_phase_z = zscore(filtered_phase)

    features: dict[str, float] = {}
    features.update(time_domain_features(raw_phase, filtered_phase_z, amp, prefix=prefix))
    features.update(spectral_features(filtered_phase_z, fs=RADAR_FS, prefix=prefix))

    return features


def extract_features_for_window(
    ids: str,
    person_id: str,
    window_id: int,
    window_start_s: float,
    window_end_s: float,
    label_hr: float,
    split: str,
) -> dict[str, float | str]:
    sample_dir = DATA_ROOT / person_id

    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample folder not found: {sample_dir}")

    matrix, range_grid = load_radar_mat(sample_dir)

    num_bins, num_time_original = matrix.shape

    matrix_10s = crop_radar_window(matrix, window_id)
    num_time_used = matrix_10s.shape[1]

    selected_bin, selected_distance, quality = choose_range_bin_by_snr(
        matrix_10s,
        range_grid,
    )

    row: dict[str, float | str] = {
        "ids": ids,
        "person_id": person_id,
        "window_id": int(window_id),
        "window_start_s": float(window_start_s),
        "window_end_s": float(window_end_s),
        "split": split,
        "label_hr": float(label_hr),

        "num_bins": int(num_bins),
        "num_time_original": int(num_time_original),
        "num_time_used": int(num_time_used),
        "radar_fs": float(RADAR_FS),

        "selected_bin": int(selected_bin),
        "selected_distance_m": (
            np.nan if selected_distance is None else float(selected_distance)
        ),

        "bin_quality_score": quality.get("quality_score", np.nan),
        "bin_quality_peak_hr": quality.get("quality_peak_hr", np.nan),
        "bin_quality_peak_ratio": quality.get("quality_peak_ratio", np.nan),
        "bin_quality_snr_like": quality.get("quality_snr_like", np.nan),
        "bin_quality_band_power": quality.get("quality_band_power", np.nan),
        "bin_quality_amp_mean": quality.get("quality_amp_mean", np.nan),
        "bin_quality_amp_cv": quality.get("quality_amp_cv", np.nan),
    }

    # main bin feature
    row.update(
        extract_features_for_bin(
            matrix_10s=matrix_10s,
            bin_idx=selected_bin,
            prefix="main_",
        )
    )

    # neighbor bin feature
    neighbor_peak_hrs = []
    neighbor_peak_ratios = []
    neighbor_snrs = []
    neighbor_scores = []

    for offset in range(-NEIGHBOR_BINS, NEIGHBOR_BINS + 1):
        bin_idx = selected_bin + offset

        if bin_idx < 0 or bin_idx >= num_bins:
            continue

        prefix = f"bin_{offset:+d}_"

        bin_features = extract_features_for_bin(
            matrix_10s=matrix_10s,
            bin_idx=bin_idx,
            prefix=prefix,
        )

        row.update(bin_features)

        peak_hr = bin_features.get(f"{prefix}peak_hr", np.nan)
        peak_ratio = bin_features.get(f"{prefix}peak_ratio", np.nan)
        snr_like = bin_features.get(f"{prefix}snr_like", np.nan)

        if np.isfinite(peak_hr):
            neighbor_peak_hrs.append(float(peak_hr))

        if np.isfinite(peak_ratio):
            neighbor_peak_ratios.append(float(peak_ratio))

        if np.isfinite(snr_like):
            neighbor_snrs.append(float(snr_like))

        if np.isfinite(peak_ratio) and np.isfinite(snr_like):
            neighbor_scores.append(float(np.log1p(snr_like) * (peak_ratio + 1e-6)))

    row["neighbor_peak_hr_mean"] = (
        float(np.mean(neighbor_peak_hrs)) if neighbor_peak_hrs else np.nan
    )
    row["neighbor_peak_hr_median"] = (
        float(np.median(neighbor_peak_hrs)) if neighbor_peak_hrs else np.nan
    )
    row["neighbor_peak_hr_std"] = (
        float(np.std(neighbor_peak_hrs)) if neighbor_peak_hrs else np.nan
    )

    row["neighbor_peak_ratio_mean"] = (
        float(np.mean(neighbor_peak_ratios)) if neighbor_peak_ratios else np.nan
    )
    row["neighbor_peak_ratio_max"] = (
        float(np.max(neighbor_peak_ratios)) if neighbor_peak_ratios else np.nan
    )

    row["neighbor_snr_like_mean"] = (
        float(np.mean(neighbor_snrs)) if neighbor_snrs else np.nan
    )
    row["neighbor_snr_like_max"] = (
        float(np.max(neighbor_snrs)) if neighbor_snrs else np.nan
    )

    row["neighbor_quality_score_mean"] = (
        float(np.mean(neighbor_scores)) if neighbor_scores else np.nan
    )
    row["neighbor_quality_score_max"] = (
        float(np.max(neighbor_scores)) if neighbor_scores else np.nan
    )

    # baseline 1：main peak
    row["radar_peak_hr_baseline"] = row.get("main_peak_hr", np.nan)

    # baseline 2：naive x2，Retain as a feature
    main_peak = row["radar_peak_hr_baseline"]

    if np.isfinite(main_peak):
        doubled = float(main_peak * 2)

        if main_peak < 65 and 45 <= doubled <= 150:
            row["radar_naive_x2_candidate"] = doubled
        else:
            row["radar_naive_x2_candidate"] = float(main_peak)
    else:
        row["radar_naive_x2_candidate"] = np.nan

    # baseline 3residual：neighbor median
    row["radar_neighbor_median_baseline"] = row["neighbor_peak_hr_median"]

    # baseline 4：smart candidate
    row["radar_smart_hr_candidate"] = smart_hr_candidate(row)

    # 兼容旧字段名：让 x2_candidate 等于 smart candidate
    row["radar_peak_hr_x2_candidate"] = row["radar_smart_hr_candidate"]

    # baseline errors，Only for observation purposes, the training code will exclude abs_error later
    baseline_names = [
        "radar_peak_hr_baseline",
        "radar_naive_x2_candidate",
        "radar_neighbor_median_baseline",
        "radar_smart_hr_candidate",
        "radar_peak_hr_x2_candidate",
    ]

    for name in baseline_names:
        pred = row.get(name, np.nan)

        if np.isfinite(pred):
            row[f"{name}_abs_error"] = abs(float(pred) - float(label_hr))
        else:
            row[f"{name}_abs_error"] = np.nan

    row["baseline_abs_error"] = row["radar_peak_hr_baseline_abs_error"]

    return row


# ============================================================
# 7. Main
# ============================================================


# Deterministic sequential numerical runtime
# ============================================================

from .preprocessing import SignalRuntime, array_key, batch_filter, batch_unwrap_scalar_detrend, batch_welch

_REFERENCE_PHASE = extract_phase_signal
_REFERENCE_ZSCORE = zscore


class OriginalExtractor:
    """Original extractor for one loaded complex radar window.

    Range-bin signals are batched for unwrap/filter/Welch and spectral
    postprocessing. Detrending intentionally remains scalar because SciPy's
    axis-batched detrend changes the floating-point operation order.
    """

    def __init__(self, runtime: SignalRuntime):
        self.runtime = runtime
        self._install_exact_runtime()

    def phase(self, matrix: np.ndarray, bin_index: int) -> np.ndarray:
        key = (array_key(matrix), int(bin_index), "unwrap-angle", "linear-detrend")
        return self.runtime.cache.get_or_compute("common_phase", key, lambda: _REFERENCE_PHASE(matrix, bin_index), (matrix,))

    def filtered(self, x: np.ndarray, fs: float = RADAR_FS, low_hz: float = LOW_HZ, high_hz: float = HIGH_HZ) -> np.ndarray:
        key = (array_key(x), float(fs), float(low_hz), float(high_hz), 4, "sosfiltfilt")
        def compute():
            if len(x) < 20: return np.asarray(x, float)
            high = min(high_hz, 0.5*fs*0.95)
            if high <= low_hz: return np.asarray(x, float)
            return batch_filter(np.asarray(x, float)[None, :], self.runtime.butter_sos(4, fs, low_hz, high))[0]
        return self.runtime.cache.get_or_compute("common_bandpass", key, compute, (x,))

    def normalized(self, x: np.ndarray) -> np.ndarray:
        return self.runtime.cache.get_or_compute("original_zscore", (array_key(x),), lambda: _REFERENCE_ZSCORE(x), (x,))

    def welch(self, x: np.ndarray, fs: float = RADAR_FS):
        key = (array_key(x), float(fs), len(x), 0, PSD_NFFT, "constant", "density")
        def compute():
            frequency, power = batch_welch(np.asarray(x, float)[None, :], fs, len(x), 0, PSD_NFFT)
            return frequency, power[0]
        return self.runtime.cache.get_or_compute("original_welch", key, compute, (x,))

    @staticmethod
    def _spectral_post(frequency: np.ndarray, psd: np.ndarray) -> list[dict[str, float]]:
        mask = (frequency >= LOW_HZ) & (frequency <= HIGH_HZ)
        bf = frequency[mask]; bp = psd[:, mask]
        totals = np.sum(bp, axis=1) + 1e-12
        centroids = np.sum(bp*bf[None, :], axis=1)/totals
        bandwidths = np.sqrt(np.sum(bp*(bf[None, :]-centroids[:, None])**2, axis=1)/totals)
        output=[]
        for row in range(bp.shape[0]):
            result={f"top{i}_{suffix}":np.nan for i in range(1,TOP_K+1) for suffix in ("hr","hz","power","ratio")}
            result.update({key:np.nan for key in ("peak_hr","peak_hz","peak_power","band_power","peak_ratio","centroid_hz","centroid_hr","bandwidth_hz","snr_like","top1_top2_power_ratio","top1_top2_hr_diff")})
            peaks,_=signal.find_peaks(bp[row]);candidates=peaks[np.argsort(bp[row,peaks])[::-1]] if len(peaks) else np.argsort(bp[row])[::-1];selected=[]
            for index in candidates:
                if all(abs(bf[index]-bf[prior])>=0.08 for prior in selected):selected.append(int(index))
                if len(selected)>=TOP_K:break
            if len(selected)<TOP_K:
                for index in np.argsort(bp[row])[::-1]:
                    if int(index) not in selected:selected.append(int(index))
                    if len(selected)>=TOP_K:break
            for rank,index in enumerate(selected[:TOP_K],1):
                hz=float(bf[index]);power=float(bp[row,index]);result[f"top{rank}_hr"]=hz*60;result[f"top{rank}_hz"]=hz;result[f"top{rank}_power"]=power;result[f"top{rank}_ratio"]=power/float(totals[row])
            result.update({"peak_hr":result["top1_hr"],"peak_hz":result["top1_hz"],"peak_power":result["top1_power"],"peak_ratio":result["top1_ratio"],"band_power":float(totals[row]),"centroid_hz":float(centroids[row]),"centroid_hr":float(centroids[row]*60),"bandwidth_hz":float(bandwidths[row])})
            exclusion=np.abs(bf-result["peak_hz"])>0.10;noise=float(np.mean(bp[row,exclusion] if np.any(exclusion) else bp[row])+1e-12);result["snr_like"]=float(result["peak_power"]/noise)
            if np.isfinite(result["top2_power"]):result["top1_top2_power_ratio"]=float(result["top1_power"]/(result["top2_power"]+1e-12));result["top1_top2_hr_diff"]=float(abs(result["top1_hr"]-result["top2_hr"]))
            output.append(result)
        return output

    def choose(self, matrix: np.ndarray, range_grid: np.ndarray | None):
        bins=np.asarray(get_candidate_bins(matrix,range_grid),int)
        phase=batch_unwrap_scalar_detrend(matrix,bins)
        for row,bin_index in enumerate(bins):
            key=(array_key(matrix),int(bin_index),"unwrap-angle","linear-detrend");self.runtime.cache.store("common_phase",key,phase[row],(matrix,phase))
        filtered=batch_filter(phase,self.runtime.butter_sos(4,RADAR_FS,LOW_HZ,HIGH_HZ))
        for row in range(len(bins)):
            key=(array_key(phase[row]),float(RADAR_FS),float(LOW_HZ),float(HIGH_HZ),4,"sosfiltfilt");self.runtime.cache.store("common_bandpass",key,filtered[row],(phase,filtered))
        normalized=np.vstack([self.normalized(row) for row in filtered])
        frequency,psd=batch_welch(normalized,RADAR_FS,normalized.shape[1],0,PSD_NFFT)
        for row in range(len(bins)):
            key=(array_key(normalized[row]),float(RADAR_FS),normalized.shape[1],0,PSD_NFFT,"constant","density");self.runtime.cache.store("original_welch",key,(frequency,psd[row]),(normalized,psd))
        specs=self._spectral_post(frequency,psd);amplitude=np.abs(matrix[bins]);amp_mean=np.mean(amplitude,axis=1);amp_std=np.std(amplitude,axis=1)
        snr=np.asarray([item["snr_like"] for item in specs]);ratio=np.asarray([item["peak_ratio"] for item in specs]);score=np.log1p(snr)*(ratio+1e-6)*np.log1p(amp_mean);peak_hr=np.asarray([item["peak_hr"] for item in specs]);score=np.where(peak_hr<LOW_PEAK_PENALTY_HR,score*0.65,score) if PENALIZE_LOW_PEAK_FOR_BIN_SELECTION else score;band_power=np.asarray([item["band_power"] for item in specs]);score=np.where((~np.isfinite(band_power))|(band_power<=1e-12),score*0.1,score)
        qualities=[{"bin_idx":float(bin_index),"quality_score":float(score[row]),"quality_peak_hr":float(specs[row]["peak_hr"]),"quality_peak_ratio":float(specs[row]["peak_ratio"]),"quality_snr_like":float(specs[row]["snr_like"]),"quality_band_power":float(specs[row]["band_power"]),"quality_amp_mean":float(amp_mean[row]),"quality_amp_cv":float(amp_std[row]/(amp_mean[row]+1e-12))} for row,bin_index in enumerate(bins)]
        if not qualities:raise ValueError("No candidate range bins")
        best=sorted(qualities,key=lambda item:item["quality_score"],reverse=True)[0];selected=int(best["bin_idx"]);distance=float(range_grid[selected]) if range_grid is not None and selected<len(range_grid) else None
        return selected,distance,best

    def _install_exact_runtime(self) -> None:
        global extract_phase_signal, bandpass_filter, zscore, compute_welch_psd, choose_range_bin_by_snr
        extract_phase_signal=self.phase;bandpass_filter=self.filtered;zscore=self.normalized;compute_welch_psd=self.welch;choose_range_bin_by_snr=self.choose

