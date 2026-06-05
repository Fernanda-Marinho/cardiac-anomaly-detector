"""
features.py – Extração de features por batimento
==================================================
Para cada pico R detectado, extrai:
  • Features de intervalo RR (temporais)
  • Features morfológicas do segmento QRS
  • Features de variabilidade (HRV local)
  • Features estatísticas do batimento (skewness, kurtosis)

Essas features alimentam o classificador de arritmias.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────

# Janela em torno do pico R para extrair o batimento (ms)
BEAT_WINDOW_PRE_MS = 100   # 100 ms antes do pico R
BEAT_WINDOW_POST_MS = 200  # 200 ms após  o pico R

# Número fixo de amostras por batimento (para vetor de morfologia uniforme)
BEAT_LENGTH_NORM = 90  # resample para 90 pontos (~250 ms a 360 Hz)


def _normalize_beat(beat: np.ndarray, length: int = BEAT_LENGTH_NORM) -> np.ndarray:
    """Interpola o segmento do batimento para comprimento fixo."""
    x_old = np.linspace(0, 1, len(beat))
    x_new = np.linspace(0, 1, length)
    return np.interp(x_new, x_old, beat)


def extract_beat_features(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: int,
    include_morphology: bool = True,
) -> np.ndarray:
    """
    Extrai matriz de features N×D para N batimentos detectados.

    Features por batimento (colunas):
    ─────────────────────────────────
    [0]  rr_pre_ms       – intervalo RR com o batimento anterior (ms)
    [1]  rr_post_ms      – intervalo RR com o próximo batimento (ms)
    [2]  rr_ratio        – rr_pre / rr_post (detecta extrassístoles)
    [3]  rr_mean_ms      – média dos 5 RR ao redor
    [4]  rr_std_ms       – desvio dos 5 RR ao redor
    [5]  rr_norm_pre     – rr_pre normalizado pela média local
    [6]  qrs_amplitude   – amplitude pico-a-pico do QRS
    [7]  qrs_energy      – energia do segmento (soma dos quadrados)
    [8]  beat_mean       – média do segmento do batimento
    [9]  beat_std        – desvio padrão do segmento
    [10] beat_skew       – assimetria do segmento (skewness)
    [11] beat_kurt       – curtose do segmento
    [12] r_amplitude     – amplitude absoluta no pico R
    [13] qrs_width_est   – estimativa da largura do QRS (cruzamentos de zero)
    [14..14+N] morph     – vetor de morfologia normalizado (se include_morphology)

    Parâmetros
    ----------
    signal             : ECG (bandpass ou bruto)
    r_peaks            : posições dos picos R (amostras)
    fs                 : frequência de amostragem
    include_morphology : inclui vetor de morfologia normalizado

    Retorna
    -------
    np.ndarray shape (n_beats, n_features)
    """
    n_beats = len(r_peaks)
    pre_samples = int(BEAT_WINDOW_PRE_MS / 1000.0 * fs)
    post_samples = int(BEAT_WINDOW_POST_MS / 1000.0 * fs)

    # Intervalos RR em ms
    rr_ms = np.diff(r_peaks) / fs * 1000.0

    base_features = 14
    n_morph = BEAT_LENGTH_NORM if include_morphology else 0
    n_features = base_features + n_morph

    X = np.zeros((n_beats, n_features), dtype=np.float32)

    for i, r_idx in enumerate(r_peaks):
        # ── Features de intervalo RR ──────────────────────────────
        rr_pre = rr_ms[i - 1] if i > 0 else rr_ms[0] if len(rr_ms) else 1000.0
        rr_post = rr_ms[i] if i < len(rr_ms) else rr_ms[-1] if len(rr_ms) else 1000.0

        # Média local de 5 intervalos ao redor
        lo_rr = max(0, i - 2)
        hi_rr = min(len(rr_ms), i + 3)
        local_rr = rr_ms[lo_rr:hi_rr] if len(rr_ms) else np.array([1000.0])
        rr_mean = float(np.mean(local_rr)) if len(local_rr) else 1000.0
        rr_std = float(np.std(local_rr)) if len(local_rr) > 1 else 0.0

        rr_ratio = rr_pre / rr_post if rr_post > 0 else 1.0
        rr_norm_pre = rr_pre / rr_mean if rr_mean > 0 else 1.0

        X[i, 0] = rr_pre
        X[i, 1] = rr_post
        X[i, 2] = rr_ratio
        X[i, 3] = rr_mean
        X[i, 4] = rr_std
        X[i, 5] = rr_norm_pre

        # ── Extrai segmento do batimento ─────────────────────────
        lo = max(0, r_idx - pre_samples)
        hi = min(len(signal), r_idx + post_samples)
        beat = signal[lo:hi]

        if len(beat) < 5:
            X[i, 12] = float(signal[r_idx]) if 0 <= r_idx < len(signal) else 0.0
            continue

        # ── Features morfológicas ────────────────────────────────
        qrs_amp = float(np.max(beat) - np.min(beat))
        qrs_energy = float(np.sum(beat ** 2))
        b_mean = float(np.mean(beat))
        b_std = float(np.std(beat))
        b_skew = float(skew(beat))
        b_kurt = float(kurtosis(beat))
        r_amp = float(signal[r_idx]) if 0 <= r_idx < len(signal) else float(np.max(beat))

        # Estimativa da largura do QRS via cruzamentos de zero do segmento
        centered = beat - np.mean(beat)
        zero_cross = np.where(np.diff(np.sign(centered)))[0]
        qrs_width = float(len(zero_cross)) / fs * 1000.0  # em ms

        X[i, 6] = qrs_amp
        X[i, 7] = qrs_energy
        X[i, 8] = b_mean
        X[i, 9] = b_std
        X[i, 10] = b_skew
        X[i, 11] = b_kurt
        X[i, 12] = r_amp
        X[i, 13] = qrs_width

        # ── Morfologia normalizada ───────────────────────────────
        if include_morphology:
            morph = _normalize_beat(beat, BEAT_LENGTH_NORM)
            # Normaliza amplitude para [-1, 1]
            m_range = morph.max() - morph.min()
            if m_range > 1e-6:
                morph = (morph - morph.min()) / m_range * 2.0 - 1.0
            X[i, base_features:] = morph.astype(np.float32)

    return X


def compute_hrv_metrics(r_peaks: np.ndarray, fs: int) -> dict:
    """
    Calcula métricas de HRV (Heart Rate Variability) globais.

    Métricas retornadas:
    ────────────────────
    num_beats  – número total de batimentos
    mean_bpm   – frequência cardíaca média (BPM)
    min_bpm    – FC mínima
    max_bpm    – FC máxima
    sdnn_ms    – desvio padrão dos intervalos NN (HRV global)
    rmssd_ms   – raiz da média quadrática de diferenças consecutivas (HRV curto)
    pnn50      – % de diferenças NN > 50 ms (parasimpático)
    rr_ms      – vetor de intervalos RR em ms
    """
    if len(r_peaks) < 2:
        return {"num_beats": len(r_peaks)}

    rr_sec = np.diff(r_peaks) / fs
    rr_ms = rr_sec * 1000.0
    diff_rr = np.diff(rr_ms)

    return {
        "num_beats":  len(r_peaks),
        "mean_bpm":   float(60.0 / np.mean(rr_sec)),
        "min_bpm":    float(60.0 / np.max(rr_sec)),
        "max_bpm":    float(60.0 / np.min(rr_sec)),
        "sdnn_ms":    float(np.std(rr_ms)),
        "rmssd_ms":   float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) else 0.0,
        "pnn50":      float(np.sum(np.abs(diff_rr) > 50.0) / len(diff_rr) * 100.0)
                      if len(diff_rr) else 0.0,
        "rr_ms":      rr_ms,
    }
