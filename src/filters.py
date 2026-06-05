"""
filters.py – Estágios de filtragem Pan-Tompkins
=================================================
Cada função implementa um estágio independente do pipeline:

  1. bandpass_filter        – Butterworth 5-15 Hz (remove linha de base e EMG)
  2. derivative_filter      – derivada de 5 pontos (realça flancos do QRS)
  3. squaring               – eleva ao quadrado (todos positivos, amplifica QRS)
  4. moving_window_integration – integrador de janela deslizante (alinha pico)
  5. baseline_wander_removal – remoção de deriva de linha de base (pré-filtro)
  6. notch_filter           – rejeita 60 Hz (ruído de rede elétrica)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


# ──────────────────────────────────────────────────────────────
# 1. Remoção de deriva de linha de base (pré-processamento)
# ──────────────────────────────────────────────────────────────

def baseline_wander_removal(
    signal: np.ndarray, fs: int, cutoff: float = 0.5, order: int = 2
) -> np.ndarray:
    """
    Remove deriva de linha de base com filtro Butterworth passa-alta.

    A respiração e o movimento do eletrodo introduzem variações lentas
    (< 0.5 Hz) que deslocam a linha de base e prejudicam a detecção de QRS.

    Parâmetros
    ----------
    cutoff : float
        Frequência de corte em Hz (padrão 0.5 Hz).
    """
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high")
    return filtfilt(b, a, signal)


# ──────────────────────────────────────────────────────────────
# 2. Filtro Notch (rejeição de ruído de rede)
# ──────────────────────────────────────────────────────────────

def notch_filter(
    signal: np.ndarray, fs: int, freq: float = 60.0, quality: float = 30.0
) -> np.ndarray:
    """
    Filtro notch IIR para remover interferência de rede (60 Hz no Brasil).

    Parâmetros
    ----------
    freq    : float  – frequência da rede (60 Hz BR / 50 Hz EU)
    quality : float  – fator Q (maior Q = banda de rejeição mais estreita)
    """
    b, a = iirnotch(freq / (0.5 * fs), quality)
    return filtfilt(b, a, signal)


# ──────────────────────────────────────────────────────────────
# 3. Filtro Bandpass Pan-Tompkins (5–15 Hz)
# ──────────────────────────────────────────────────────────────

def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    lowcut: float = 5.0,
    highcut: float = 15.0,
    order: int = 2,
) -> np.ndarray:
    """
    Butterworth bandpass 5–15 Hz (Pan & Tompkins, 1985).

    Por que 5–15 Hz?
    ─────────────────
    • < 5 Hz: deriva de linha de base, respiração, ondas P e T
    • > 15 Hz: ruído muscular (EMG), harmônicos de rede elétrica
    • A energia do QRS está concentrada nesta faixa → maximiza SNR

    filtfilt = fase zero (sem atraso de tempo), ideal para análise offline.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Garante que as frequências estejam dentro do intervalo válido (0, 1)
    low = max(1e-4, min(low, 0.999))
    high = max(low + 1e-4, min(high, 0.9999))
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


# ──────────────────────────────────────────────────────────────
# 4. Filtro Derivada de 5 pontos
# ──────────────────────────────────────────────────────────────

def derivative_filter(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Derivada de 5 pontos do artigo original Pan-Tompkins.

    H(z) = (1/8T)[ -z^{-2} - 2z^{-1} + 2z + z^{2} ]
    Coeficientes: [-1, -2, 0, +2, +1] / (8T)

    Amplifica as inclinações de alta frequência dos flancos do QRS
    enquanto suprime componentes de baixa frequência (P/T).
    """
    T = 1.0 / fs
    kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) / (8.0 * T)
    return np.convolve(signal, kernel, mode="same")


# ──────────────────────────────────────────────────────────────
# 5. Elevar ao quadrado
# ──────────────────────────────────────────────────────────────

def squaring(signal: np.ndarray) -> np.ndarray:
    """
    Eleva ao quadrado ponto a ponto.

    Efeitos:
    • Torna todos os valores positivos
    • Amplifica não linearmente grandes inclinações (QRS) vs pequenas (ruído/T)
    • Prepara para integração por janela deslizante
    """
    return signal ** 2


# ──────────────────────────────────────────────────────────────
# 6. Integração por janela deslizante (Moving Window Integration)
# ──────────────────────────────────────────────────────────────

def moving_window_integration(
    signal: np.ndarray, fs: int, window_ms: int = 150
) -> np.ndarray:
    """
    Integrador de soma acumulada (box-car / janela deslizante).

    A janela de ~150 ms engloba um complexo QRS típico, alinhando
    o pico de saída com o centro do QRS.
    Janelas maiores fundem batimentos adjacentes em frequências cardíacas
    elevadas (> 180 BPM).

    Parâmetros
    ----------
    window_ms : int  – largura da janela em milissegundos (padrão 150 ms)
    """
    window = max(1, int((window_ms / 1000.0) * fs))
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


# ──────────────────────────────────────────────────────────────
# 7. Pipeline completo de pré-processamento
# ──────────────────────────────────────────────────────────────

def preprocess(
    signal: np.ndarray,
    fs: int,
    remove_baseline: bool = True,
    apply_notch: bool = True,
    notch_freq: float = 60.0,
) -> dict[str, np.ndarray]:
    """
    Executa o pipeline completo de filtragem e retorna cada estágio.

    Ordem: baseline → notch → bandpass → derivada → squaring → MWI

    Retorna
    -------
    dict com chaves: "clean", "bandpass", "derivative", "squared", "integrated"
    """
    clean = signal.copy()

    if remove_baseline:
        clean = baseline_wander_removal(clean, fs)

    if apply_notch:
        clean = notch_filter(clean, fs, freq=notch_freq)

    bp = bandpass_filter(clean, fs)
    deriv = derivative_filter(bp, fs)
    sq = squaring(deriv)
    mwi = moving_window_integration(sq, fs)

    return {
        "clean": clean,
        "bandpass": bp,
        "derivative": deriv,
        "squared": sq,
        "integrated": mwi,
    }
