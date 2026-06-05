"""
pan_tompkins.py – Detecção de picos R com limiar adaptativo
============================================================
Implementa o algoritmo completo de Pan & Tompkins (1985) incluindo:
  • Duplo limiar adaptativo (sinal + ruído)
  • Período refratário de 200 ms
  • Busca retroativa (searchback) para batimentos perdidos
  • Busca do pico verdadeiro no sinal original

Referência:
  Pan, J. & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
  IEEE Transactions on Biomedical Engineering, 32(3), 230-236.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .filters import preprocess


# ──────────────────────────────────────────────────────────────
# Limiar adaptativo (núcleo do algoritmo)
# ──────────────────────────────────────────────────────────────

def adaptive_thresholding(integrated: np.ndarray, fs: int) -> np.ndarray:
    """
    Detecção de picos R com duplo limiar adaptativo (Pan-Tompkins).

    O algoritmo original mantém dois estimadores em tempo real:
      SPKI  – nível de pico de sinal (média móvel de 8 batimentos aceitos)
      NPKI  – nível de pico de ruído (média móvel de 8 picos rejeitados)
      ThresholdI = NPKI + 0.25 × (SPKI − NPKI)

    Além disso implementa busca retroativa (searchback):
      Se nenhum pico for detectado em 166 % do RR médio esperado,
      o algoritmo busca com metade do limiar o pico mais alto no intervalo.

    Parâmetros
    ----------
    integrated : np.ndarray – sinal após MWI
    fs         : int        – frequência de amostragem

    Retorna
    -------
    np.ndarray de inteiros com as posições dos picos R aceitos
    """
    min_rr_samples = int(0.200 * fs)  # período refratário 200 ms

    # Inicialização: 2 segundos para estimar SPKI inicial
    init_samples = min(2 * fs, len(integrated))
    SPKI = np.max(integrated[:init_samples]) * 0.25
    NPKI = np.max(integrated[:init_samples]) * 0.125
    threshold = NPKI + 0.25 * (SPKI - NPKI)

    # Todos os candidatos locais acima de vizinhos
    candidates, _ = find_peaks(integrated, distance=min_rr_samples)

    accepted: list[int] = []
    rr_history: list[float] = []   # histórico dos últimos 8 intervalos RR
    last_qrs = -min_rr_samples

    for idx in candidates:
        # Período refratário estrito
        if (idx - last_qrs) < min_rr_samples:
            continue

        peak_val = integrated[idx]

        # ── Busca retroativa ──────────────────────────────────────────
        if rr_history:
            rr_avg = np.mean(rr_history[-8:])
            if (idx - last_qrs) > 1.66 * rr_avg:
                # Procura o maior pico no intervalo perdido com limiar reduzido
                search_start = last_qrs + min_rr_samples
                search_end = idx
                if search_end > search_start:
                    window = integrated[search_start:search_end]
                    local_peaks, _ = find_peaks(window)
                    if len(local_peaks):
                        best_local = local_peaks[np.argmax(window[local_peaks])]
                        local_val = window[best_local]
                        if local_val > 0.5 * threshold:
                            actual_idx = search_start + best_local
                            accepted.append(actual_idx)
                            rr_history.append(actual_idx - last_qrs)
                            last_qrs = actual_idx
                            SPKI = 0.25 * local_val + 0.75 * SPKI
                            threshold = NPKI + 0.25 * (SPKI - NPKI)
                            continue

        # ── Decisão normal ───────────────────────────────────────────
        if peak_val > threshold:
            accepted.append(idx)
            rr_history.append(idx - last_qrs)
            last_qrs = idx
            SPKI = 0.125 * peak_val + 0.875 * SPKI
        else:
            NPKI = 0.125 * peak_val + 0.875 * NPKI

        # Atualiza limiar após cada candidato
        threshold = NPKI + 0.25 * (SPKI - NPKI)

    return np.array(accepted, dtype=int)


def refine_r_peaks(
    r_peaks_mwi: np.ndarray,
    original_signal: np.ndarray,
    fs: int,
    search_window_ms: int = 100,
) -> np.ndarray:
    """
    Refina a posição dos picos R no sinal original (não no MWI).

    O MWI introduz um atraso de ~75 ms; esta função encontra o verdadeiro
    pico de amplitude máxima em uma janela ao redor de cada candidato.

    Parâmetros
    ----------
    r_peaks_mwi     : picos detectados no sinal integrado
    original_signal : sinal ECG bruto (ou pós-bandpass)
    search_window_ms: metade da janela de busca em ms
    """
    half_win = int((search_window_ms / 1000.0) * fs)
    refined: list[int] = []

    for idx in r_peaks_mwi:
        lo = max(0, idx - half_win)
        hi = min(len(original_signal), idx + half_win)
        local_seg = original_signal[lo:hi]
        # Pega o índice do máximo absoluto (QRS tem pico positivo em MLII)
        peak_offset = int(np.argmax(np.abs(local_seg)))
        refined.append(lo + peak_offset)

    return np.array(refined, dtype=int)


# ──────────────────────────────────────────────────────────────
# Pipeline Pan-Tompkins completo
# ──────────────────────────────────────────────────────────────

def pan_tompkins(
    signal: np.ndarray,
    fs: int,
    refine: bool = True,
    remove_baseline: bool = True,
    apply_notch: bool = True,
    notch_freq: float = 60.0,
) -> dict:
    """
    Pipeline Pan-Tompkins completo.

    Parâmetros
    ----------
    signal          : ECG bruto em mV
    fs              : frequência de amostragem
    refine          : se True, refina picos no sinal original pós-bandpass
    remove_baseline : remove deriva de linha de base (alta-passa 0.5 Hz)
    apply_notch     : aplica notch para ruído de rede
    notch_freq      : frequência do notch (60 Hz Brasil)

    Retorna
    -------
    dict com chaves:
        "clean"      – sinal após pré-filtros
        "bandpass"   – após filtro bandpass 5-15 Hz
        "derivative" – após derivada de 5 pontos
        "squared"    – após elevação ao quadrado
        "integrated" – após MWI
        "r_peaks"    – posições dos picos R no sinal original (amostras)
    """
    stages = preprocess(
        signal,
        fs,
        remove_baseline=remove_baseline,
        apply_notch=apply_notch,
        notch_freq=notch_freq,
    )

    r_peaks_mwi = adaptive_thresholding(stages["integrated"], fs)

    if refine and len(r_peaks_mwi):
        r_peaks = refine_r_peaks(r_peaks_mwi, stages["bandpass"], fs)
    else:
        r_peaks = r_peaks_mwi

    stages["r_peaks"] = r_peaks
    return stages
