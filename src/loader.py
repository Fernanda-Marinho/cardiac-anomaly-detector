"""
loader.py – Download e segmentação de registros MIT-BIH
========================================================
Responsabilidades:
  • Baixar registros do PhysioNet via wfdb (cache local em data/raw/)
  • Segmentar o sinal em janelas de duração configurável
  • Retornar sinal, frequência de amostragem, anotações e eixo de tempo
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import wfdb

# ──────────────────────────────────────────────────────────────
# Registros disponíveis no MIT-BIH Arrhythmia Database
# ──────────────────────────────────────────────────────────────
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

# Mapeamento dos rótulos AAMI para classe legível
AAMI_MAP = {
    # Normal
    "N": "Normal",
    "L": "LBBB",       # Left Bundle Branch Block
    "R": "RBBB",       # Right Bundle Branch Block
    "e": "APC",        # Atrial Premature Contraction (aberrante)
    "j": "NPC",        # Nodal (junctional) premature beat
    # Supraventricular ectopic (S)
    "A": "APC",        # Atrial Premature Contraction
    "a": "APC",
    "S": "SPC",        # Supraventricular premature
    "J": "NPC",
    # Ventricular ectopic (V)
    "V": "PVC",        # Premature Ventricular Contraction
    "E": "VE",         # Ventricular escape
    # Fusion (F)
    "F": "Fusion",
    # Unknown / paced
    "/": "Paced",
    "f": "Paced",
    "Q": "Unknown",
    # Artefacto / ruído
    "~": "Noise",
    "|": "IsoElectric",
    "+": "RhythmChange",
}


def list_records() -> list[str]:
    """Retorna a lista de todos os 48 registros MIT-BIH disponíveis."""
    return MITBIH_RECORDS.copy()


def load_record(
    record_name: str = "100",
    duration_sec: Optional[int] = 30,
    channel: int = 0,
    data_dir: Optional[str | Path] = None,
) -> Tuple[np.ndarray, int, wfdb.Annotation, np.ndarray]:
    """
    Carrega um registro MIT-BIH do PhysioNet (com cache local).

    Parâmetros
    ----------
    record_name : str
        ID do registro (ex: "100", "200", "208").
    duration_sec : int | None
        Segundos a carregar. None = registro completo (~30 min).
    channel : int
        Canal ECG a usar. 0 = MLII (padrão MIT-BIH), 1 = V1/V5.
    data_dir : str | Path | None
        Diretório de cache local. Se None, baixa direto do PhysioNet.

    Retorna
    -------
    signal : np.ndarray   – ECG em mV
    fs     : int          – frequência de amostragem (360 Hz)
    ann    : wfdb.Annotation – anotações do especialista
    t      : np.ndarray   – eixo de tempo em segundos
    """
    if record_name not in MITBIH_RECORDS:
        raise ValueError(
            f"Registro '{record_name}' não encontrado. "
            f"Use list_records() para ver os disponíveis."
        )

    pn_dir: Optional[str] = "mitdb"
    record_path: Optional[str] = None

    # Se existe cache local, usa ele
    if data_dir is not None:
        cache = Path(data_dir) / "raw"
        cache.mkdir(parents=True, exist_ok=True)
        local = cache / record_name
        if (local.with_suffix(".dat")).exists():
            record_path = str(local)
            pn_dir = None
            print(f"[loader] Usando cache local: {local}")
        else:
            # Baixa e salva no cache
            print(f"[loader] Baixando {record_name} do PhysioNet MIT-BIH …")
            wfdb.dl_database("mitdb", str(cache), records=[record_name])
            record_path = str(local)
            pn_dir = None

    # Carrega registro
    record = wfdb.rdrecord(
        record_path if record_path else record_name,
        pn_dir=pn_dir,
        sampto=None,
    )

    fs: int = record.fs  # 360 Hz
    sampto = duration_sec * fs if duration_sec else record.sig_len

    # Garante que não ultrapassa o tamanho do sinal
    sampto = min(sampto, record.sig_len)

    signal: np.ndarray = record.p_signal[:sampto, channel]

    # Substitui NaN por interpolação linear
    nan_mask = np.isnan(signal)
    if nan_mask.any():
        signal[nan_mask] = np.interp(
            np.flatnonzero(nan_mask),
            np.flatnonzero(~nan_mask),
            signal[~nan_mask],
        )

    # Anotações
    ann = wfdb.rdann(
        record_path if record_path else record_name,
        "atr",
        pn_dir=pn_dir,
        sampto=sampto,
    )

    t = np.arange(sampto) / fs

    print(
        f"[loader] {record_name} | {sampto} amostras | "
        f"fs={fs} Hz | {sampto/fs:.1f} s | {len(ann.sample)} anotações"
    )
    return signal, fs, ann, t


def get_aami_label(symbol: str) -> str:
    """Converte símbolo MIT-BIH para rótulo AAMI legível."""
    return AAMI_MAP.get(symbol, "Unknown")


def export_csv(
    signal: np.ndarray,
    t: np.ndarray,
    r_peaks: np.ndarray,
    labels: list[str],
    output_path: str | Path,
) -> None:
    """
    Exporta o sinal + picos R + rótulos para CSV em data/processed/.

    Colunas: time_s, ecg_mv, is_r_peak, beat_label
    """
    import pandas as pd

    is_r = np.zeros(len(signal), dtype=int)
    label_col = np.full(len(signal), "", dtype=object)

    for i, idx in enumerate(r_peaks):
        if 0 <= idx < len(signal):
            is_r[idx] = 1
            if i < len(labels):
                label_col[idx] = labels[i]

    df = pd.DataFrame(
        {"time_s": t, "ecg_mv": signal, "is_r_peak": is_r, "beat_label": label_col}
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[loader] CSV exportado → {output_path}")
