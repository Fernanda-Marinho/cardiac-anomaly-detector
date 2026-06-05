"""
plotter.py – Visualização multi-painel do pipeline ECG
=======================================================
Gera figuras PNG em alta resolução com tema escuro (estilo Dracula):
  • 7 painéis: ECG bruto, bandpass, derivada, squared, MWI, RR tachogram, espectro
  • Anotações de referência MIT-BIH sobrepostas
  • Picos R detectados marcados com círculos
  • Rótulos de arritmia coloridos por classe
  • Matriz de confusão para avaliação do classificador
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless – sem janela gráfica
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import welch

# ──────────────────────────────────────────────────────────────
# Paleta de cores (tema Dracula)
# ──────────────────────────────────────────────────────────────
PALETTE = {
    "bg":         "#0d1117",
    "panel":      "#161b22",
    "border":     "#30363d",
    "text":       "#c9d1d9",
    "subtext":    "#8b949e",
    "raw":        "#8be9fd",
    "bandpass":   "#50fa7b",
    "derivative": "#ffb86c",
    "squared":    "#ff79c6",
    "integrated": "#bd93f9",
    "rr_line":    "#f1fa8c",
    "rr_mean":    "#ff79c6",
    "ref_ann":    "#ff5555",
    "r_peak":     "#f8f8f2",
    "spectrum":   "#6272a4",
    # Cores por classe de arritmia
    "N":     "#50fa7b",   # verde – normal
    "V":     "#ff5555",   # vermelho – PVC
    "A":     "#ffb86c",   # laranja – APC
    "L":     "#8be9fd",   # ciano – LBBB
    "R":     "#bd93f9",   # lilás – RBBB
    "/":     "#f1fa8c",   # amarelo – paced
    "F":     "#ff79c6",   # rosa – fusion
    "TACHY": "#ffb86c",
    "BRADY": "#8be9fd",
    "PAUSE": "#ff5555",
}

ARRHYTHMIA_NAMES = {
    "N": "Normal", "V": "PVC", "A": "APC", "L": "LBBB",
    "R": "RBBB", "/": "Paced", "F": "Fusion",
    "TACHY": "Taquicardia", "BRADY": "Bradicardia", "PAUSE": "Pausa",
}


def _style_ax(ax: plt.Axes, title: str, ylabel: str = "Amplitude") -> None:
    """Aplica estilo escuro em um Axes."""
    ax.set_facecolor(PALETTE["panel"])
    ax.set_title(title, color=PALETTE["text"], fontsize=10, pad=4)
    ax.set_ylabel(ylabel, color=PALETTE["subtext"], fontsize=8)
    ax.tick_params(colors=PALETTE["subtext"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])


def plot_results(
    t: np.ndarray,
    raw: np.ndarray,
    stages: dict[str, np.ndarray],
    r_peaks: np.ndarray,
    ann,                       # wfdb.Annotation
    metrics: dict,
    beat_labels: Optional[list[str]] = None,
    rhythm_info: Optional[dict] = None,
    output_path: str | Path = "ecg_pan_tompkins.png",
) -> None:
    """
    Gera figura de 7 painéis com o pipeline completo.

    Painéis:
    --------
    1. ECG bruto + anotações de referência MIT-BIH
    2. Bandpass 5-15 Hz
    3. Derivada de 5 pontos
    4. Sinal ao quadrado
    5. MWI + picos R detectados (coloridos por classe)
    6. RR tachogram com HRV
    7. Espectro de potência do ECG
    """
    n_panels = 7
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(18, 24), facecolor=PALETTE["bg"]
    )

    # ── Título principal ────────────────────────────────────────
    rhythm_str = ""
    if rhythm_info:
        rhythm_str = (
            f"  |  {rhythm_info.get('rhythm', '')}  "
            f"|  {rhythm_info.get('mean_bpm', 0):.0f} BPM"
        )
    fig.suptitle(
        f"Pan-Tompkins QRS Detection — MIT-BIH{rhythm_str}",
        color=PALETTE["text"], fontsize=14, y=0.995,
    )

    # ── Painel 1: ECG bruto ─────────────────────────────────────
    ax1 = axes[0]
    _style_ax(ax1, "1. ECG Bruto (mV)", "mV")
    ax1.plot(t, raw, color=PALETTE["raw"], linewidth=0.7, alpha=0.9)

    # Sobrepõe anotações de referência do MIT-BIH
    if ann is not None:
        ref_s = ann.sample[ann.sample < len(raw)]
        ref_sym = ann.symbol[:len(ref_s)]
        for samp, sym in zip(ref_s, ref_sym):
            color = PALETTE.get(sym, PALETTE["ref_ann"])
            ax1.axvline(t[samp], color=color, alpha=0.35, linewidth=0.8)
            y_pos = raw[samp] + 0.15 * (raw.max() - raw.min())
            ax1.text(t[samp], y_pos, sym,
                     color=color, fontsize=5, ha="center", va="bottom")

    # ── Painéis 2-5: estágios de filtragem ─────────────────────
    stage_cfg = [
        ("bandpass",   PALETTE["bandpass"],   "2. Bandpass 5–15 Hz"),
        ("derivative", PALETTE["derivative"], "3. Derivada 5 pontos"),
        ("squared",    PALETTE["squared"],    "4. Sinal ao Quadrado"),
        ("integrated", PALETTE["integrated"], "5. MWI + Picos R"),
    ]

    for ax, (key, color, title) in zip(axes[1:5], stage_cfg):
        _style_ax(ax, title)
        sig = stages.get(key, np.zeros_like(t))
        ax.plot(t, sig, color=color, linewidth=0.7, alpha=0.9)

    # Marca picos R no painel MWI (painel índice 4)
    ax5 = axes[4]
    integ = stages.get("integrated", np.zeros_like(t))
    if len(r_peaks):
        # Colore picos por classe de arritmia
        if beat_labels and len(beat_labels) == len(r_peaks):
            for idx, lbl in zip(r_peaks, beat_labels):
                if 0 <= idx < len(t):
                    c = PALETTE.get(lbl, PALETTE["r_peak"])
                    ax5.plot(t[idx], integ[idx], "o",
                             color=c, markersize=5, zorder=5, alpha=0.9)
        else:
            ax5.plot(t[r_peaks], integ[r_peaks], "o",
                     color=PALETTE["r_peak"], markersize=5,
                     label="Picos R", zorder=5)

    # Legenda de classes detectadas
    if beat_labels:
        unique_lbls = set(beat_labels)
        patches = [
            mpatches.Patch(
                color=PALETTE.get(l, "#888"),
                label=ARRHYTHMIA_NAMES.get(l, l),
            )
            for l in sorted(unique_lbls)
        ]
        ax5.legend(
            handles=patches,
            facecolor=PALETTE["bg"],
            labelcolor=PALETTE["text"],
            fontsize=7,
            loc="upper right",
        )

    # ── Painel 6: RR Tachogram ──────────────────────────────────
    ax6 = axes[5]
    _style_ax(ax6, "6. RR Tachogram (HRV)", "RR (ms)")
    rr_ms = metrics.get("rr_ms", np.array([]))
    if len(rr_ms):
        beat_n = np.arange(1, len(rr_ms) + 1)
        ax6.plot(beat_n, rr_ms, color=PALETTE["rr_line"], linewidth=1.2)
        rr_mean = float(np.mean(rr_ms))
        ax6.axhline(
            rr_mean, color=PALETTE["rr_mean"], linestyle="--", linewidth=1,
            label=f"Média {rr_mean:.0f} ms  "
                  f"SDNN={metrics.get('sdnn_ms', 0):.1f} ms  "
                  f"RMSSD={metrics.get('rmssd_ms', 0):.1f} ms"
        )
        ax6.fill_between(beat_n, rr_ms, rr_mean,
                         alpha=0.12, color=PALETTE["rr_line"])
        ax6.legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["text"],
                   fontsize=8)
    ax6.set_xlabel("Número do batimento", color=PALETTE["subtext"], fontsize=8)

    # ── Painel 7: Espectro de Potência ─────────────────────────
    ax7 = axes[6]
    _style_ax(ax7, "7. Espectro de Potência (ECG bruto)", "PSD (mV²/Hz)")
    fs = round(1.0 / (t[1] - t[0])) if len(t) > 1 else 360
    freqs, psd = welch(raw, fs=fs, nperseg=min(1024, len(raw) // 4))
    ax7.semilogy(freqs, psd, color=PALETTE["spectrum"], linewidth=1.0)
    ax7.axvspan(5, 15, alpha=0.15, color=PALETTE["bandpass"],
                label="Banda QRS (5-15 Hz)")
    ax7.set_xlim(0, min(80, freqs[-1]))
    ax7.set_xlabel("Frequência (Hz)", color=PALETTE["subtext"], fontsize=8)
    ax7.legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["text"], fontsize=8)

    # ── Rótulo de tempo nos painéis ECG ─────────────────────────
    for ax in axes[:6]:
        ax.set_xlabel("Tempo (s)", color=PALETTE["subtext"], fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.997])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"[plotter] Figura salva → {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str | Path = "confusion_matrix.png",
    normalize: bool = True,
) -> None:
    """
    Plota e salva a matriz de confusão.

    Parâmetros
    ----------
    y_true      : rótulos verdadeiros (inteiros)
    y_pred      : rótulos preditos (inteiros)
    class_names : nomes das classes
    normalize   : normaliza por linha (proporção)
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_plot = cm.astype(float) / row_sums
    else:
        cm_plot = cm.astype(float)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)),
                           facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    im = ax.imshow(cm_plot, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right",
                       color=PALETTE["text"], fontsize=9)
    ax.set_yticklabels(class_names, color=PALETTE["text"], fontsize=9)

    thresh = cm_plot.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            txt = f"{val:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if val > thresh else PALETTE["text"])

    ax.set_xlabel("Predito", color=PALETTE["text"], fontsize=10)
    ax.set_ylabel("Real", color=PALETTE["text"], fontsize=10)
    ax.set_title("Matriz de Confusão" + (" (normalizada)" if normalize else ""),
                 color=PALETTE["text"], fontsize=12)

    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    ax.tick_params(colors=PALETTE["subtext"])

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"[plotter] Matriz de confusão salva → {output_path}")
