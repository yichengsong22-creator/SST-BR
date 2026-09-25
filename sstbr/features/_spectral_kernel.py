"""Current-window multi-band filterbank numerical kernel."""
from __future__ import annotations
import math
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal
from .preprocessing import SignalRuntime, array_key, batch_filter

HR_LOW_BPM = 45.0
HR_HIGH_BPM = 150.0
LOW_HZ = HR_LOW_BPM / 60.0
HIGH_HZ = HR_HIGH_BPM / 60.0
WINDOW_SECONDS = 5.0
FILTERBANK_K_LIST = [3, 4]
VMD_SIGNAL_SOURCE_PRIORITY = ["rt_tracked_final_bin", "erg_final_candidate_bin", "energy_weighted_phase", "selected_bin"]
MAX_WEIGHTED_PHASE_BINS = 8
PSD_NFFT = 4096
HAVE_VMDPY = False
VMDPY_VMD = None
ALLOW_FILTERBANK_FALLBACK = True
AVMD_TAU = 0.0
AVMD_DC = 0
AVMD_INIT = 1
AVMD_TOL = 1e-5

# Quality is log(1+SNR) * peak_ratio * log(1+energy); peaks below 65 BPM use factor 0.75.

def safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default



def parse_bin_list(x) -> List[int]:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    out = []
    for part in re.split(r"[|,; ]+", s):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except Exception:
            continue
    return sorted(set(out))


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

def phase_signal(matrix_10s: np.ndarray, bin_idx: int) -> np.ndarray:
    bin_idx = int(bin_idx)
    iq = matrix_10s[bin_idx, :]
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


def compute_psd_1d(x: np.ndarray, fs: float, nfft: int = PSD_NFFT) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

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


def spectral_summary_1d(x: np.ndarray, fs: float) -> Dict[str, float]:
    out = {
        "peak_hr": np.nan,
        "peak_hz": np.nan,
        "peak_power": np.nan,
        "band_power": np.nan,
        "peak_ratio": np.nan,
        "snr_like": np.nan,
        "centroid_hr": np.nan,
        "bandwidth_hz": np.nan,
        "top1_top2_ratio": np.nan,
        "top1_top2_hr_diff": np.nan,
    }

    freqs, psd = compute_psd_1d(x, fs=fs, nfft=PSD_NFFT)
    if len(freqs) == 0:
        return out

    band_mask = (freqs >= LOW_HZ) & (freqs <= min(HIGH_HZ, fs * 0.5 * 0.95))
    if not np.any(band_mask):
        return out

    bf = freqs[band_mask]
    bp = psd[band_mask]
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
    })

    if len(order) >= 2:
        i2 = int(order[1])
        f2 = float(bf[i2])
        p2 = float(bp[i2])
        out["top1_top2_ratio"] = float(p1 / (p2 + 1e-12))
        out["top1_top2_hr_diff"] = float(abs(f1 - f2) * 60.0)

    return out


def select_phase_signal_for_avmd(row: pd.Series, matrix_10s: np.ndarray) -> Tuple[np.ndarray, Dict[str, float | str]]:
    n_bins = matrix_10s.shape[0]

    def valid_bin(v) -> Optional[int]:
        vv = safe_float(v, default=np.nan)
        if np.isfinite(vv):
            b = int(round(vv))
            if 0 <= b < n_bins:
                return b
        return None

    # Prefer tracked or energy-supported phase sources before fallback signals
    # are considered.
    # Source 1: tracked bin.
    if "rt_tracked_final_bin" in row.index and "rt_tracked_final_bin" in VMD_SIGNAL_SOURCE_PRIORITY:
        b = valid_bin(row.get("rt_tracked_final_bin"))
        if b is not None:
            x = phase_signal(matrix_10s, b)
            return x, {"avmd_signal_source": "rt_tracked_final_bin", "avmd_signal_bin": int(b)}

    # Source 2: energy final candidate bin.
    if "erg_final_candidate_bin" in row.index and "erg_final_candidate_bin" in VMD_SIGNAL_SOURCE_PRIORITY:
        b = valid_bin(row.get("erg_final_candidate_bin"))
        if b is not None:
            x = phase_signal(matrix_10s, b)
            return x, {"avmd_signal_source": "erg_final_candidate_bin", "avmd_signal_bin": int(b)}

    # Source 3: energy-weighted phase from selected bins.
    if "energy_weighted_phase" in VMD_SIGNAL_SOURCE_PRIORITY:
        bins = parse_bin_list(row.get("erg_selected_bins", ""))
        bins = [b for b in bins if 0 <= b < n_bins]

        if bins:
            # Use top bins by amplitude energy within this window.
            energy = np.sum(np.abs(matrix_10s[bins, :]), axis=1)
            order = np.argsort(energy)[::-1]
            bins = [bins[i] for i in order[:MAX_WEIGHTED_PHASE_BINS]]
            energy = energy[order[:MAX_WEIGHTED_PHASE_BINS]]
            weights = energy / (np.sum(energy) + 1e-12)

            signals = []
            for b in bins:
                signals.append(robust_zscore(phase_signal(matrix_10s, b)))

            x = np.sum(np.vstack(signals) * weights.reshape(-1, 1), axis=0)
            return x, {
                "avmd_signal_source": "energy_weighted_phase",
                "avmd_signal_bin": np.nan,
                "avmd_signal_weighted_bins": "|".join(map(str, bins)),
                "avmd_signal_weighted_bin_count": int(len(bins)),
            }

    # Source 4: original selected_bin.
    if "selected_bin" in row.index and "selected_bin" in VMD_SIGNAL_SOURCE_PRIORITY:
        b = valid_bin(row.get("selected_bin"))
        if b is not None:
            x = phase_signal(matrix_10s, b)
            return x, {"avmd_signal_source": "selected_bin", "avmd_signal_bin": int(b)}

    # Last fallback: max amplitude bin.
    amp_energy = np.sum(np.abs(matrix_10s), axis=1)
    b = int(np.argmax(amp_energy))
    x = phase_signal(matrix_10s, b)
    return x, {"avmd_signal_source": "fallback_max_amp_bin", "avmd_signal_bin": int(b)}


# ============================================================
# 6. Adaptive VMD / fallback decomposition
# ============================================================

def run_vmdpy_decomposition(x: np.ndarray, K: int, alpha: float) -> Optional[np.ndarray]:
    if not HAVE_VMDPY or VMDPY_VMD is None:
        return None

    try:
        # vmdpy expects a 1D array.
        u, _, _ = VMDPY_VMD(
            x,
            alpha,
            AVMD_TAU,
            K,
            AVMD_DC,
            AVMD_INIT,
            AVMD_TOL,
        )
        # u shape normally [K, N]
        u = np.asarray(u, dtype=float)
        if u.ndim != 2:
            return None
        if u.shape[0] != K and u.shape[1] == K:
            u = u.T
        return u
    except Exception:
        return None


def run_filterbank_decomposition(x: np.ndarray, fs: float, K: int) -> np.ndarray:
    """
    Conservative fallback when vmdpy is not installed.
    It is not true VMD, but it creates mode-like narrowband signals.
    """
    edges = np.linspace(LOW_HZ, HIGH_HZ, K + 1)
    modes = []

    for i in range(K):
        lo = edges[i]
        hi = edges[i + 1]
        # Slightly overlap bands to avoid cutting peaks at boundaries.
        bw = hi - lo
        lo2 = max(0.10, lo - 0.10 * bw)
        hi2 = min(0.5 * fs * 0.95, hi + 0.10 * bw)

        if hi2 <= lo2:
            modes.append(np.zeros_like(x))
            continue

        try:
            sos = signal.butter(3, [lo2 / (0.5 * fs), hi2 / (0.5 * fs)], btype="bandpass", output="sos")
            mode = signal.sosfiltfilt(sos, x)
        except Exception:
            mode = np.zeros_like(x)

        modes.append(np.asarray(mode, dtype=float))

    return np.vstack(modes)


def evaluate_modes(modes: np.ndarray, x: np.ndarray, fs: float, K: int, alpha: float, backend: str) -> Dict[str, float | str]:
    """
    Select the best mode according to HR-band spectral quality.
    """
    if modes is None or modes.ndim != 2 or modes.shape[0] == 0:
        return {
            "avmd_backend": backend,
            "avmd_success": 0,
            "avmd_selected_K": K,
            "avmd_selected_alpha": alpha,
            "avmd_selected_mode": np.nan,
            "avmd_candidate_hr": np.nan,
            "avmd_candidate_snr": np.nan,
            "avmd_candidate_peak_ratio": np.nan,
        }

    # Rank candidate modes using cardiac-band spectral quality.
    mode_rows = []
    for i in range(modes.shape[0]):
        mode = np.asarray(modes[i], dtype=float)
        spec = spectral_summary_1d(mode, fs)
        mode_energy = float(np.sum(mode ** 2))
        score = 0.0

        if np.isfinite(spec["peak_hr"]):
            score = math.log1p(max(spec["snr_like"], 0.0)) * max(spec["peak_ratio"], 0.0)
            score *= math.log1p(max(mode_energy, 0.0))

            # weak penalty for low peaks that may be respiratory/half-frequency artifacts
            if spec["peak_hr"] < 65.0:
                score *= 0.75

        mode_rows.append({
            "mode_idx": int(i),
            "mode_energy": mode_energy,
            "score": float(score),
            **spec,
        })

    mdf = pd.DataFrame(mode_rows)

    if len(mdf) == 0 or not np.isfinite(mdf["score"]).any():
        best_idx = 0
    else:
        best_idx = int(mdf["score"].astype(float).idxmax())

    best = mdf.loc[best_idx].to_dict()

    mode_hrs = pd.to_numeric(mdf["peak_hr"], errors="coerce").to_numpy(dtype=float)
    candidate_hr = safe_float(best.get("peak_hr"), default=np.nan)

    if np.isfinite(candidate_hr):
        agree5 = int(np.sum(np.abs(mode_hrs - candidate_hr) <= 5.0))
        agree8 = int(np.sum(np.abs(mode_hrs - candidate_hr) <= 8.0))
    else:
        agree5 = 0
        agree8 = 0

    signal_energy = float(np.sum(np.asarray(x, dtype=float) ** 2) + 1e-12)
    recon_energy = float(np.sum(np.asarray(modes, dtype=float) ** 2))

    out: Dict[str, float | str] = {
        "avmd_backend": backend,
        "avmd_success": 1,
        "avmd_selected_K": int(K),
        "avmd_selected_alpha": float(alpha),
        "avmd_selected_mode": int(best.get("mode_idx", 0)),

        "avmd_candidate_hr": candidate_hr,
        "avmd_candidate_hz": safe_float(best.get("peak_hz"), default=np.nan),
        "avmd_candidate_snr": safe_float(best.get("snr_like"), default=np.nan),
        "avmd_candidate_peak_ratio": safe_float(best.get("peak_ratio"), default=np.nan),
        "avmd_candidate_band_power": safe_float(best.get("band_power"), default=np.nan),
        "avmd_candidate_peak_power": safe_float(best.get("peak_power"), default=np.nan),
        "avmd_candidate_centroid_hr": safe_float(best.get("centroid_hr"), default=np.nan),
        "avmd_candidate_bandwidth_hz": safe_float(best.get("bandwidth_hz"), default=np.nan),
        "avmd_candidate_top1_top2_ratio": safe_float(best.get("top1_top2_ratio"), default=np.nan),
        "avmd_candidate_top1_top2_hr_diff": safe_float(best.get("top1_top2_hr_diff"), default=np.nan),
        "avmd_candidate_mode_energy": safe_float(best.get("mode_energy"), default=np.nan),
        "avmd_candidate_quality_score": safe_float(best.get("score"), default=np.nan),

        "avmd_recon_energy_ratio": float(recon_energy / signal_energy),
        "avmd_mode_agreement_count_5bpm": agree5,
        "avmd_mode_agreement_count_8bpm": agree8,
    }

    out.update(finite_stats(mode_hrs, "avmd_mode_hr"))

    out["avmd_candidate_x2_if_low"] = (
        float(candidate_hr * 2.0)
        if np.isfinite(candidate_hr) and candidate_hr < 65.0 and candidate_hr * 2.0 <= HR_HIGH_BPM
        else candidate_hr
    )
    out["avmd_candidate_half_if_high"] = (
        float(candidate_hr / 2.0)
        if np.isfinite(candidate_hr) and candidate_hr > 120.0 and candidate_hr / 2.0 >= HR_LOW_BPM
        else candidate_hr
    )

    return out


def extract_adaptive_vmd_features(x_raw: np.ndarray, fs: float) -> Dict[str, float | str]:
    x = np.asarray(x_raw, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = signal.detrend(x) if len(x) > 10 else x - np.mean(x)
    x = bandpass(x, fs)
    x = robust_zscore(x)

    base_spec = spectral_summary_1d(x, fs)

    best_features: Optional[Dict[str, float | str]] = None
    best_score = -np.inf

    backend_used = "none"

    if HAVE_VMDPY:
        backend_used = "vmdpy"
        for K in AVMD_K_LIST:
            for alpha in AVMD_ALPHA_LIST:
                modes = run_vmdpy_decomposition(x, K=K, alpha=alpha)
                if modes is None:
                    continue
                feat = evaluate_modes(modes, x=x, fs=fs, K=K, alpha=alpha, backend="vmdpy")
                score = safe_float(feat.get("avmd_candidate_quality_score"), default=-np.inf)
                if score > best_score:
                    best_score = score
                    best_features = feat

    if best_features is None:
        if not ALLOW_FILTERBANK_FALLBACK:
            best_features = {
                "avmd_backend": "failed_no_vmdpy",
                "avmd_success": 0,
                "avmd_selected_K": np.nan,
                "avmd_selected_alpha": np.nan,
                "avmd_selected_mode": np.nan,
                "avmd_candidate_hr": np.nan,
                "avmd_candidate_snr": np.nan,
                "avmd_candidate_peak_ratio": np.nan,
            }
        else:
            backend_used = "filterbank_fallback"
            for K in FILTERBANK_K_LIST:
                modes = run_filterbank_decomposition(x, fs=fs, K=K)
                feat = evaluate_modes(modes, x=x, fs=fs, K=K, alpha=np.nan, backend="filterbank_fallback")
                score = safe_float(feat.get("avmd_candidate_quality_score"), default=-np.inf)
                if score > best_score:
                    best_score = score
                    best_features = feat

    if best_features is None:
        best_features = {
            "avmd_backend": "failed_unknown",
            "avmd_success": 0,
            "avmd_candidate_hr": np.nan,
        }

    best_features.update({
        "avmd_signal_std": float(np.std(x)),
        "avmd_signal_ptp": float(np.ptp(x)),
        "avmd_signal_base_peak_hr": base_spec["peak_hr"],
        "avmd_signal_base_snr": base_spec["snr_like"],
        "avmd_signal_base_peak_ratio": base_spec["peak_ratio"],
        "avmd_backend_used_overall": backend_used,
    })

    return best_features


# ============================================================
# 7. Per-row and 30s fusion features
# ============================================================

def process_one_row(row: pd.Series, matrix_10s: np.ndarray, fs: float) -> Dict[str, float | str]:
    x_raw, signal_meta = select_phase_signal_for_avmd(row, matrix_10s)
    avmd_feat = extract_adaptive_vmd_features(x_raw, fs=fs)

    out: Dict[str, float | str] = {}
    out.update(signal_meta)
    out.update(avmd_feat)

    # Differences with existing radar candidates.
    cand = safe_float(out.get("avmd_candidate_hr"), default=np.nan)
    for col in ["radar_smart_hr_candidate", "erg_final_candidate_hr", "rt_final_peak_hr", "radar_peak_hr_baseline"]:
        if col in row.index:
            v = safe_float(row.get(col), default=np.nan)
            out[f"avmd_candidate_minus_{col}"] = float(cand - v) if np.isfinite(cand) and np.isfinite(v) else np.nan

    out["avmd_error"] = ""
    return out



_REFERENCE_PHASE = phase_signal
_REFERENCE_ROBUST_ZSCORE = robust_zscore

class FilterbankExtractor:
    """VMD-inspired fallback filterbank with precomputed third-order SOS."""

    def __init__(self,runtime:SignalRuntime):
        self.runtime=runtime;self._install()

    def phase(self,matrix:np.ndarray,bin_idx:int)->np.ndarray:
        key=(array_key(matrix),int(bin_idx),"unwrap-angle","linear-detrend")
        return self.runtime.cache.get_or_compute("common_phase",key,lambda:_REFERENCE_PHASE(matrix,bin_idx),(matrix,))

    def filtered(self,x:np.ndarray,fs:float,low_hz:float=LOW_HZ,high_hz:float=HIGH_HZ)->np.ndarray:
        key=(array_key(x),float(fs),float(low_hz),float(high_hz),4,"sosfiltfilt")
        return self.runtime.cache.get_or_compute("common_bandpass",key,lambda:batch_filter(np.asarray(x,float)[None,:],self.runtime.butter_sos(4,fs,low_hz,min(high_hz,.5*fs*.95)))[0],(x,))

    def normalized(self,x:np.ndarray)->np.ndarray:
        return self.runtime.cache.get_or_compute("filterbank_zscore",(array_key(x),),lambda:_REFERENCE_ROBUST_ZSCORE(x),(x,))

    def decomposition(self,x:np.ndarray,fs:float,K:int)->np.ndarray:
        edges=np.linspace(LOW_HZ,HIGH_HZ,K+1);modes=[]
        for index in range(K):
            low,high=edges[index],edges[index+1];width=high-low;low=max(.10,low-.10*width);high=min(.5*fs*.95,high+.10*width)
            modes.append(batch_filter(np.asarray(x,float)[None,:],self.runtime.butter_sos(3,fs,low,high))[0] if high>low else np.zeros_like(x))
        return np.vstack(modes)

    def _install(self):
        global HAVE_VMDPY,ALLOW_FILTERBANK_FALLBACK,phase_signal,bandpass,robust_zscore,run_filterbank_decomposition
        HAVE_VMDPY=False;ALLOW_FILTERBANK_FALLBACK=True;phase_signal=self.phase;bandpass=self.filtered;robust_zscore=self.normalized;run_filterbank_decomposition=self.decomposition

