"""
Pan-Tompkins QRS Detection on MIT-BIH Arrhythmia Database
==========================================================
Requirements:
    pip install wfdb numpy scipy matplotlib

Usage:
    python pan_tompkins_mitbih.py

The script downloads record 100 from MIT-BIH directly via the wfdb library
(no manual download required) and runs a full Pan-Tompkins pipeline on it.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless – saves to file instead of a window
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import wfdb

# ─────────────────────────────────────────────
# 1.  LOAD DATA FROM PHYSIONET (MIT-BIH)
# ─────────────────────────────────────────────

def load_mitbih_record(record_name: str = "100", duration_sec: int = 30):
    """
    Download and return one record from the MIT-BIH Arrhythmia Database.

    Parameters
    ----------
    record_name : str
        Record ID.  MIT-BIH has records: 100–124, 200–234 (48 total).
        Start with '100' – a clean normal sinus rhythm recording.
    duration_sec : int
        How many seconds to load (max ~1800 for a full 30-min record).

    Returns
    -------
    signal : np.ndarray   – raw ECG (mV), MLII channel
    fs     : int          – sampling frequency in Hz (360 for MIT-BIH)
    ann    : wfdb.Annotation – doctor annotations (beat labels + positions)
    t      : np.ndarray   – time axis in seconds
    """
    print(f"Downloading record {record_name} from PhysioNet MIT-BIH …")

    # wfdb.rdrecord fetches .hea + .dat from PhysioNet automatically.
    # sampfrom / sampto select a window (samples, not seconds).
    record = wfdb.rdrecord(
        record_name,
        pn_dir="mitdb",          # database folder on PhysioNet
        sampto=None              # None = full record
    )

    fs = record.fs               # 360 Hz for all MIT-BIH records
    sampto = duration_sec * fs

    # Channel 0 is MLII (standard limb lead) for most MIT-BIH records
    signal = record.p_signal[:sampto, 0]

    # Load reference annotations (expert-labelled R-peak positions)
    ann = wfdb.rdann(
        record_name,
        "atr",
        pn_dir="mitdb",
        sampto=sampto
    )

    t = np.arange(len(signal)) / fs

    print(f"  Loaded {len(signal)} samples  |  fs={fs} Hz  |  "
          f"duration={len(signal)/fs:.1f} s")
    print(f"  Reference beats in window: {len(ann.sample)}")
    return signal, fs, ann, t


# ─────────────────────────────────────────────
# 2.  PAN-TOMPKINS PIPELINE
# ─────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, fs: int,
                    lowcut: float = 5.0, highcut: float = 15.0,
                    order: int = 2) -> np.ndarray:
    """
    Butterworth bandpass 5–15 Hz.

    Why 5–15 Hz?
      - Below 5 Hz: baseline wander, respiration artefacts, P/T waves
      - Above 15 Hz: muscle (EMG) noise, powerline harmonics
      - QRS energy is concentrated in this band → maximises SNR
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    # filtfilt = zero-phase (no time delay), better than lfilter for offline analysis
    return filtfilt(b, a, signal)


def derivative_filter(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    5-point derivative from the original Pan-Tompkins paper.

    H(z) = (1/8T)[-z^{-2} - 2z^{-1} + 2z + z^{2}]

    Amplifies high-frequency slopes (QRS flanks) while suppressing
    low-frequency components (P/T waves).
    """
    T = 1.0 / fs
    # coefficients: [-1, -2, 0, +2, +1] * 1/(8T)
    kernel = np.array([-1, -2, 0, 2, 1]) / (8.0 * T)
    return np.convolve(signal, kernel, mode="same")


def squaring(signal: np.ndarray) -> np.ndarray:
    """
    Point-wise squaring.
      - Makes all values positive
      - Non-linearly amplifies large slopes (QRS) vs small ones (noise/T-wave)
    """
    return signal ** 2


def moving_window_integration(signal: np.ndarray, fs: int,
                               window_ms: int = 150) -> np.ndarray:
    """
    Running-sum (box-car) integrator.

    Window ~150 ms encompasses a typical QRS complex width so the
    output peak aligns with the QRS centre.  Wider windows merge
    adjacent beats at high heart rates.
    """
    window = int((window_ms / 1000.0) * fs)
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def adaptive_thresholding(integrated: np.ndarray, fs: int) -> np.ndarray:
    """
    Adaptive dual-threshold peak detection (simplified Pan-Tompkins).

    The original paper maintains two running estimates:
      SPKI  – signal peak level (8-beat running mean of accepted QRS peaks)
      NPKI  – noise peak level  (8-beat running mean of rejected peaks)
      ThresholdI = NPKI + 0.25 * (SPKI - NPKI)

    This implementation uses a two-pass strategy:
      Pass 1 – find all local maxima above a global seed threshold
      Pass 2 – apply refractory period (200 ms) and adaptive update

    Returns indices of accepted R-peaks in the *original signal* domain.
    """
    min_rr_samples = int(0.200 * fs)   # 200 ms refractory period

    # Seed: threshold starts at 50 % of signal mean
    SPKI = np.mean(integrated)
    NPKI = 0.0
    threshold = NPKI + 0.25 * (SPKI - NPKI)

    # Find all candidate peaks (must be higher than neighbours)
    candidates, _ = find_peaks(integrated, distance=min_rr_samples)

    accepted = []
    last_qrs = -min_rr_samples  # allow first beat anywhere

    for idx in candidates:
        peak_val = integrated[idx]

        if (idx - last_qrs) < min_rr_samples:
            continue  # still in refractory period

        if peak_val > threshold:
            accepted.append(idx)
            last_qrs = idx
            # Update signal estimate (running mean, weight 1/8)
            SPKI = 0.125 * peak_val + 0.875 * SPKI
        else:
            # Update noise estimate
            NPKI = 0.125 * peak_val + 0.875 * NPKI

        # Recalculate threshold after each candidate
        threshold = NPKI + 0.25 * (SPKI - NPKI)

    return np.array(accepted, dtype=int)


def pan_tompkins(signal: np.ndarray, fs: int) -> dict:
    """
    Full Pan-Tompkins pipeline.  Returns a dict with every intermediate
    signal so you can inspect / plot each stage.
    """
    bp      = bandpass_filter(signal, fs)
    deriv   = derivative_filter(bp, fs)
    sq      = squaring(deriv)
    mwi     = moving_window_integration(sq, fs)
    r_peaks = adaptive_thresholding(mwi, fs)

    return {
        "bandpass":    bp,
        "derivative":  deriv,
        "squared":     sq,
        "integrated":  mwi,
        "r_peaks":     r_peaks,
    }


# ─────────────────────────────────────────────
# 3.  HEART RATE & HRV METRICS
# ─────────────────────────────────────────────

def compute_hr_metrics(r_peaks: np.ndarray, fs: int) -> dict:
    if len(r_peaks) < 2:
        return {}

    rr_sec   = np.diff(r_peaks) / fs          # RR intervals in seconds
    rr_ms    = rr_sec * 1000.0                 # in milliseconds

    return {
        "num_beats":  len(r_peaks),
        "mean_bpm":   60.0 / np.mean(rr_sec),
        "min_bpm":    60.0 / np.max(rr_sec),
        "max_bpm":    60.0 / np.min(rr_sec),
        "sdnn_ms":    np.std(rr_ms),           # HRV: SD of NN intervals
        "rmssd_ms":   np.sqrt(np.mean(np.diff(rr_ms) ** 2)),  # HRV: RMSSD
        "rr_ms":      rr_ms,
    }


# ─────────────────────────────────────────────
# 4.  PLOTTING
# ─────────────────────────────────────────────

def plot_results(t, raw, stages, r_peaks, ann, metrics,
                 output_path="ecg_pan_tompkins.png"):
    """
    6-panel figure:
      1. Raw ECG with reference annotations
      2. Bandpass filtered
      3. Derivative
      4. Squared
      5. Moving-window integrated + detected R-peaks
      6. RR-interval tachogram (HRV)
    """
    fig, axes = plt.subplots(6, 1, figsize=(16, 20), facecolor="#0d1117")
    fig.suptitle("Pan-Tompkins QRS Detection — MIT-BIH Record 100",
                 color="white", fontsize=15, y=0.99)

    plot_cfg = [
        (raw,                    "#8be9fd", "1. Raw ECG (mV)"),
        (stages["bandpass"],     "#50fa7b", "2. Bandpass Filtered (5–15 Hz)"),
        (stages["derivative"],   "#ffb86c", "3. Derivative"),
        (stages["squared"],      "#ff79c6", "4. Squared"),
        (stages["integrated"],   "#bd93f9", "5. MWI + Detected R-peaks"),
    ]

    for ax, (sig, color, title) in zip(axes[:5], plot_cfg):
        ax.set_facecolor("#161b22")
        ax.plot(t, sig, color=color, linewidth=0.8, alpha=0.9)
        ax.set_title(title, color="white", fontsize=10, pad=4)
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Annotate raw ECG with reference labels from MIT-BIH
    ref_samples = ann.sample[ann.sample < len(raw)]
    ref_symbols = ann.symbol[:len(ref_samples)]
    for samp, sym in zip(ref_samples, ref_symbols):
        axes[0].axvline(t[samp], color="#ff5555", alpha=0.4, linewidth=0.8)
        axes[0].text(t[samp], raw[samp] + 0.2, sym,
                     color="#ff5555", fontsize=6, ha="center")

    # Mark detected R-peaks on integrated signal
    axes[4].plot(t[r_peaks], stages["integrated"][r_peaks],
                 "o", color="#f8f8f2", markersize=5, label="Detected R-peaks",
                 zorder=5)
    axes[4].legend(facecolor="#0d1117", labelcolor="white", fontsize=8)

    # ── Panel 6: RR tachogram ──────────────────────────────────────────
    ax6 = axes[5]
    ax6.set_facecolor("#161b22")
    rr_ms  = metrics.get("rr_ms", np.array([]))
    beat_n = np.arange(1, len(rr_ms) + 1)
    if len(rr_ms):
        ax6.plot(beat_n, rr_ms, color="#f1fa8c", linewidth=1.2)
        ax6.axhline(np.mean(rr_ms), color="#ff79c6", linestyle="--",
                    linewidth=1, label=f"Mean {np.mean(rr_ms):.0f} ms")
        ax6.fill_between(beat_n, rr_ms, np.mean(rr_ms),
                         alpha=0.15, color="#f1fa8c")
        ax6.legend(facecolor="#0d1117", labelcolor="white", fontsize=8)
    ax6.set_title("6. RR-Interval Tachogram (HRV)", color="white",
                  fontsize=10, pad=4)
    ax6.set_xlabel("Beat number", color="#888", fontsize=9)
    ax6.set_ylabel("RR interval (ms)", color="#888", fontsize=9)
    ax6.tick_params(colors="#888", labelsize=8)
    for spine in ax6.spines.values():
        spine.set_edgecolor("#30363d")

    # Shared x-label for ECG panels
    for ax in axes[:5]:
        ax.set_xlabel("Time (s)", color="#888", fontsize=8)
        ax.set_ylabel("Amplitude", color="#888", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    print(f"Plot saved → {output_path}")


# ─────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────

def main():
    # ── Load ──────────────────────────────────
    # Change record_name to try other records, e.g. "200" (arrhythmias),
    # "208" (PVCs), "214" (LBBB)
    raw, fs, ann, t = load_mitbih_record(record_name="100", duration_sec=30)

    # ── Pan-Tompkins ──────────────────────────
    print("\nRunning Pan-Tompkins pipeline …")
    stages   = pan_tompkins(raw, fs)
    r_peaks  = stages["r_peaks"]

    # ── Metrics ───────────────────────────────
    metrics  = compute_hr_metrics(r_peaks, fs)

    print("\n── Results ─────────────────────────────────")
    print(f"  Detected beats  : {metrics['num_beats']}")
    print(f"  Mean heart rate : {metrics['mean_bpm']:.1f} BPM")
    print(f"  HR range        : {metrics['min_bpm']:.1f} – {metrics['max_bpm']:.1f} BPM")
    print(f"  SDNN            : {metrics['sdnn_ms']:.1f} ms  (HRV)")
    print(f"  RMSSD           : {metrics['rmssd_ms']:.1f} ms (HRV)")

    # Compare with reference annotations
    ref_r = ann.sample[ann.symbol == "N"]   # 'N' = normal beat
    ref_r = ref_r[ref_r < len(raw)]
    print(f"\n  Reference normal beats : {len(ref_r)}")

    # Simple TP/FP/FN with 50 ms tolerance
    tol = int(0.050 * fs)
    tp = sum(any(abs(r_peaks - ref) <= tol) for ref in ref_r)
    fp = len(r_peaks) - tp
    fn = len(ref_r) - tp
    sens  = tp / (tp + fn) if (tp + fn) else 0
    ppv   = tp / (tp + fp) if (tp + fp) else 0
    print(f"  True Positives         : {tp}")
    print(f"  False Positives        : {fp}")
    print(f"  False Negatives        : {fn}")
    print(f"  Sensitivity            : {sens * 100:.1f} %")
    print(f"  Positive Predictive    : {ppv * 100:.1f} %")
    print("────────────────────────────────────────────\n")

    # ── Plot ──────────────────────────────────
    plot_results(t, raw, stages, r_peaks, ann, metrics)


if __name__ == "__main__":
    main()