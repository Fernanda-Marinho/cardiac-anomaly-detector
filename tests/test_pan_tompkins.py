"""
tests/test_pan_tompkins.py – Testes unitários pytest
=====================================================
Testa cada módulo do pipeline de forma isolada, sem acesso ao PhysioNet.
Usa sinais sintéticos para garantir reprodutibilidade.

Executar:
    cd cardiac-anomaly-detector
    pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.filters import (
    bandpass_filter,
    derivative_filter,
    squaring,
    moving_window_integration,
    baseline_wander_removal,
    notch_filter,
    preprocess,
)
from src.pan_tompkins import adaptive_thresholding, refine_r_peaks, pan_tompkins
from src.features import extract_beat_features, compute_hrv_metrics
from src.metrics import evaluate_detection, detection_report, classification_report_beats

FS = 360  # MIT-BIH sampling rate


# ──────────────────────────────────────────────────────────────
# Fixtures – sinais sintéticos
# ──────────────────────────────────────────────────────────────

def _synthetic_ecg(
    n_beats: int = 10,
    fs: int = FS,
    bpm: float = 70.0,
    noise_amp: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera ECG sintético com QRS gaussianos.

    Retorna (signal, r_peak_positions).
    """
    rng = np.random.default_rng(seed)
    rr_samples = int(fs * 60.0 / bpm)
    duration = n_beats * rr_samples + fs  # +1 s de buffer
    signal = rng.normal(0, noise_amp, duration)

    r_positions = []
    for k in range(n_beats):
        center = int(fs * 0.5) + k * rr_samples  # offset de 0.5 s
        r_positions.append(center)
        # Pico R gaussiano com amplitude ~1.5 mV e duração ~50 ms
        sigma = int(0.025 * fs)
        for offset in range(-3 * sigma, 3 * sigma + 1):
            idx = center + offset
            if 0 <= idx < duration:
                signal[idx] += 1.5 * np.exp(-0.5 * (offset / sigma) ** 2)

    return signal, np.array(r_positions, dtype=int)


# ──────────────────────────────────────────────────────────────
# Testes de filters.py
# ──────────────────────────────────────────────────────────────

class TestFilters:
    def test_bandpass_output_shape(self):
        sig = np.random.randn(FS * 10)
        out = bandpass_filter(sig, FS)
        assert out.shape == sig.shape

    def test_bandpass_attenuates_dc(self):
        """Sinal DC deve ser removido pelo bandpass."""
        sig = np.ones(FS * 5)
        out = bandpass_filter(sig, FS)
        assert np.abs(out).max() < 0.01

    def test_derivative_output_shape(self):
        sig = np.random.randn(FS * 5)
        out = derivative_filter(sig, FS)
        assert out.shape == sig.shape

    def test_derivative_of_constant_near_zero(self):
        """Derivada de constante deve ser ~zero."""
        sig = np.ones(FS * 5) * 2.0
        out = derivative_filter(sig, FS)
        # Bordas têm artefactos de convolução; ignora 5 amostras das bordas
        assert np.abs(out[5:-5]).max() < 1e-6

    def test_squaring_nonnegative(self):
        sig = np.random.randn(FS * 5)
        out = squaring(sig)
        assert (out >= 0).all()

    def test_squaring_amplifies_large(self):
        sig = np.array([0.1, 0.5, 1.0, 2.0])
        out = squaring(sig)
        assert out[3] > out[2] > out[1] > out[0]

    def test_mwi_output_shape(self):
        sig = np.random.randn(FS * 5)
        out = moving_window_integration(sig, FS)
        assert out.shape == sig.shape

    def test_mwi_smooths_impulse(self):
        """MWI deve suavizar um impulso único."""
        sig = np.zeros(FS)
        sig[FS // 2] = 1.0
        out = moving_window_integration(sig, FS, window_ms=150)
        # Energia deve estar espalhada, não concentrada em um ponto
        assert out.max() < 1.0

    def test_baseline_removal_reduces_dc(self):
        t = np.arange(FS * 10) / FS
        drift = 0.5 * np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz = linha de base
        ecg = np.random.randn(len(t)) * 0.1 + drift
        out = baseline_wander_removal(ecg, FS)
        # A deriva deve estar muito reduzida
        assert np.std(out) < np.std(ecg)

    def test_notch_filter_output_shape(self):
        sig = np.random.randn(FS * 5)
        out = notch_filter(sig, FS, freq=60.0)
        assert out.shape == sig.shape

    def test_preprocess_returns_all_keys(self):
        sig, _ = _synthetic_ecg()
        stages = preprocess(sig, FS)
        for key in ("clean", "bandpass", "derivative", "squared", "integrated"):
            assert key in stages
            assert stages[key].shape == sig.shape


# ──────────────────────────────────────────────────────────────
# Testes de pan_tompkins.py
# ──────────────────────────────────────────────────────────────

class TestPanTompkins:
    def test_detects_all_beats_clean_signal(self):
        """Pan-Tompkins deve detectar ≥90% dos batimentos em sinal limpo."""
        sig, ref = _synthetic_ecg(n_beats=15, noise_amp=0.01)
        stages = pan_tompkins(sig, FS)
        r_peaks = stages["r_peaks"]
        tol = int(0.05 * FS)
        tp = sum(any(abs(r_peaks - r) <= tol) for r in ref)
        sensitivity = tp / len(ref)
        assert sensitivity >= 0.90, f"Sensibilidade {sensitivity:.2%} < 90%"

    def test_pan_tompkins_returns_dict_with_r_peaks(self):
        sig, _ = _synthetic_ecg(n_beats=8)
        result = pan_tompkins(sig, FS)
        assert "r_peaks" in result
        assert isinstance(result["r_peaks"], np.ndarray)

    def test_r_peaks_within_signal_bounds(self):
        sig, _ = _synthetic_ecg(n_beats=10)
        stages = pan_tompkins(sig, FS)
        r_peaks = stages["r_peaks"]
        assert (r_peaks >= 0).all()
        assert (r_peaks < len(sig)).all()

    def test_r_peaks_respect_refractory_period(self):
        """Nenhum par de picos deve estar a menos de 200 ms (período refratário)."""
        sig, _ = _synthetic_ecg(n_beats=12, bpm=70.0)
        stages = pan_tompkins(sig, FS)
        r_peaks = stages["r_peaks"]
        if len(r_peaks) > 1:
            min_rr = np.min(np.diff(r_peaks)) / FS * 1000.0
            assert min_rr >= 180.0, f"RR mínimo {min_rr:.1f} ms < 180 ms"

    def test_refine_r_peaks_stays_close(self):
        """Refinamento não deve deslocar picos mais de 100 ms."""
        sig, ref = _synthetic_ecg(n_beats=10)
        bp = bandpass_filter(sig, FS)
        refined = refine_r_peaks(ref, bp, FS, search_window_ms=100)
        displacements = np.abs(refined - ref) / FS * 1000.0
        assert displacements.max() <= 100.0


# ──────────────────────────────────────────────────────────────
# Testes de features.py
# ──────────────────────────────────────────────────────────────

class TestFeatures:
    def test_feature_matrix_shape(self):
        sig, r_peaks = _synthetic_ecg(n_beats=8)
        X = extract_beat_features(sig, r_peaks, FS, include_morphology=False)
        assert X.shape == (8, 14)

    def test_feature_matrix_with_morphology(self):
        sig, r_peaks = _synthetic_ecg(n_beats=8)
        X = extract_beat_features(sig, r_peaks, FS, include_morphology=True)
        # 14 base + 90 morfologia = 104
        assert X.shape[1] == 14 + 90

    def test_features_are_finite(self):
        sig, r_peaks = _synthetic_ecg(n_beats=10)
        X = extract_beat_features(sig, r_peaks, FS)
        assert np.isfinite(X).all(), "Features contêm NaN ou Inf"

    def test_rr_features_positive(self):
        sig, r_peaks = _synthetic_ecg(n_beats=10)
        X = extract_beat_features(sig, r_peaks, FS)
        # Colunas 0-1: rr_pre_ms, rr_post_ms devem ser positivos
        assert (X[:, 0] > 0).all()
        assert (X[:, 1] > 0).all()

    def test_hrv_metrics_keys(self):
        _, r_peaks = _synthetic_ecg(n_beats=10)
        hrv = compute_hrv_metrics(r_peaks, FS)
        for key in ("num_beats", "mean_bpm", "sdnn_ms", "rmssd_ms", "rr_ms"):
            assert key in hrv

    def test_hrv_bpm_range(self):
        _, r_peaks = _synthetic_ecg(n_beats=10, bpm=70.0)
        hrv = compute_hrv_metrics(r_peaks, FS)
        assert 60.0 <= hrv["mean_bpm"] <= 80.0, f"BPM fora do esperado: {hrv['mean_bpm']:.1f}"

    def test_hrv_single_beat_no_crash(self):
        r_peaks = np.array([100])
        hrv = compute_hrv_metrics(r_peaks, FS)
        assert hrv["num_beats"] == 1


# ──────────────────────────────────────────────────────────────
# Testes de metrics.py
# ──────────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_detection(self):
        ref = np.array([100, 460, 820, 1180])
        det = ref.copy()
        res = evaluate_detection(det, ref, FS, tolerance_ms=50)
        assert res["tp"] == 4
        assert res["fp"] == 0
        assert res["fn"] == 0
        assert abs(res["sensitivity"] - 1.0) < 1e-9
        assert abs(res["ppv"] - 1.0) < 1e-9

    def test_all_missed(self):
        ref = np.array([100, 460, 820])
        det = np.array([], dtype=int)
        res = evaluate_detection(det, ref, FS)
        assert res["tp"] == 0
        assert res["fn"] == 3
        assert res["sensitivity"] == 0.0

    def test_all_false_positives(self):
        ref = np.array([100, 460, 820])
        det = np.array([50, 400, 700])   # todos fora da tolerância
        res = evaluate_detection(det, ref, FS, tolerance_ms=10)
        assert res["tp"] == 0
        assert res["ppv"] == 0.0

    def test_f1_score_calculation(self):
        ref = np.arange(0, 10) * FS  # 10 batimentos a cada 1 s
        det = np.arange(0, 8) * FS   # detecta apenas 8 (FN=2, FP=0)
        res = evaluate_detection(det, ref, FS)
        assert res["tp"] == 8
        assert res["fn"] == 2
        assert res["fp"] == 0
        expected_se = 8 / 10
        assert abs(res["sensitivity"] - expected_se) < 1e-9

    def test_detection_report_string(self):
        res = {"tp": 5, "fp": 1, "fn": 0,
               "sensitivity": 1.0, "ppv": 0.833, "f1": 0.909,
               "tolerance_ms": 50, "n_detected": 6, "n_reference": 5}
        report = detection_report(res, record_name="100", verbose=False)
        assert "Sensibilidade" in report
        assert "100.00 %" in report

    def test_classification_report_beats(self):
        y_true = ["N", "N", "V", "A", "N", "V"]
        y_pred = ["N", "N", "V", "N", "N", "V"]
        result = classification_report_beats(y_true, y_pred, verbose=False)
        assert "per_class" in result
        assert "macro_f1" in result
        assert result["per_class"]["V"]["tp"] == 2
        assert result["per_class"]["A"]["tp"] == 0

    def test_tolerance_matters(self):
        """Tolerância maior deve dar mais TPs."""
        ref = np.array([100, 460])
        det = np.array([108, 470])  # ~22 ms de erro
        res_tight = evaluate_detection(det, ref, FS, tolerance_ms=10)
        res_loose = evaluate_detection(det, ref, FS, tolerance_ms=50)
        assert res_loose["tp"] >= res_tight["tp"]
