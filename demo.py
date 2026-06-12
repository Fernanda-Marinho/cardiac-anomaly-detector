"""
demo_arrhythmia.py – Demonstração com arritmias sintéticas forçadas
====================================================================
Gera um ECG sintético com arritmias injetadas manualmente e roda
o pipeline completo (Pan-Tompkins + classificador).

Arritmias simuladas
-------------------
  Batimentos 1–4   : Normal  (ritmo sinusal, FC ~70 BPM)
  Batimento  5     : PVC     (extrassístole ventricular – QRS largo, prematuro)
  Batimentos 6–7   : Normal
  Batimento  8     : APC     (extrassístole atrial – QRS estreito, prematuro)
  Batimentos 9–10  : Normal
  Batimentos 11–14 : Taquicardia (FC ~130 BPM)
  Batimento  15    : Pausa sinusal (RR > 2 s)
  Batimentos 16–19 : Normal
  Batimentos 20–24 : FA simulada (RR irregulares, CV > 15%)

Uso:
    python demo_arrhythmia.py
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.filters import preprocess
from src.pan_tompkins import adaptive_thresholding, refine_r_peaks
from src.features import extract_beat_features, compute_hrv_metrics
from src.classifier import BeatClassifier
from src.plotter import PALETTE
from src.metrics import evaluate_detection, detection_report

FS = 360  # Hz – igual ao MIT-BIH
OUTPUT_DIR = Path(".")

# ──────────────────────────────────────────────────────────────
# 1. Gerador de formas de onda individuais
# ──────────────────────────────────────────────────────────────

def _gaussian_qrs(
    n_samples: int,
    center: int,
    amplitude: float = 1.5,
    width_ms: float = 50.0,
    fs: int = FS,
) -> np.ndarray:
    """Pico R gaussiano estreito (batimento normal)."""
    sigma = int((width_ms / 1000.0) * fs / 2.5)
    seg = np.zeros(n_samples)
    for i in range(n_samples):
        seg[i] += amplitude * np.exp(-0.5 * ((i - center) / sigma) ** 2)
    return seg


def _pvc_waveform(
    n_samples: int,
    center: int,
    amplitude: float = 2.0,
    width_ms: float = 160.0,   # QRS largo – característica do PVC
    fs: int = FS,
) -> np.ndarray:
    """
    Forma de onda de PVC: QRS largo bifásico.
    Pico positivo seguido de deflexão negativa (padrão ventricular).
    """
    sigma = int((width_ms / 1000.0) * fs / 2.5)
    seg = np.zeros(n_samples)
    for i in range(n_samples):
        d = i - center
        # Componente positiva (R) + componente negativa (S amplo)
        seg[i] += amplitude * np.exp(-0.5 * (d / sigma) ** 2)
        seg[i] -= 0.6 * amplitude * np.exp(-0.5 * ((d - sigma) / (sigma * 0.8)) ** 2)
    return seg


def _t_wave(
    n_samples: int,
    center: int,
    amplitude: float = 0.3,
    fs: int = FS,
) -> np.ndarray:
    """Onda T suave após o QRS."""
    sigma = int(0.08 * fs)  # ~80 ms
    seg = np.zeros(n_samples)
    t_center = center + int(0.25 * fs)  # T wave ~250 ms após R
    for i in range(n_samples):
        seg[i] += amplitude * np.exp(-0.5 * ((i - t_center) / sigma) ** 2)
    return seg


# ──────────────────────────────────────────────────────────────
# 2. Construção do ECG sintético com arritmias
# ──────────────────────────────────────────────────────────────

def build_arrhythmia_ecg(fs: int = FS) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Constrói ECG sintético de ~30 s com as seguintes arritmias:

    Retorna
    -------
    signal     : np.ndarray  – ECG em mV
    r_true     : np.ndarray  – posições verdadeiras dos picos R
    true_labels: list[str]   – rótulo AAMI de cada batimento
    """
    rng = np.random.default_rng(42)
    duration_s = 30
    total = duration_s * fs
    signal = rng.normal(0, 0.025, total)   # ruído de fundo leve

    # Sequência de batimentos com RR em ms
    # Formato: (tipo, rr_ms_até_este_batimento)
    beats_plan: list[tuple[str, float]] = [
        # ── Sinusal normal (70 BPM ≈ 857 ms) ──────────────
        ("N",      857),
        ("N",      860),
        ("N",      855),
        ("N",      858),
        # ── PVC no batimento 5 (prematuro + QRS largo) ────
        # RR prematuro: 60% do normal
        ("V",      515),
        # Pausa compensatória pós-PVC (RR longo)
        ("N",     1200),
        ("N",      860),
        # ── APC no batimento 8 (prematuro, QRS estreito) ──
        ("A",      560),
        ("N",      900),
        ("N",      858),
        # ── Taquicardia (batimentos 11–14, FC ~130 BPM) ───
        ("TACHY",  462),
        ("TACHY",  460),
        ("TACHY",  465),
        ("TACHY",  458),
        # ── Pausa sinusal (RR > 2000 ms) ──────────────────
        ("PAUSE", 2200),
        # ── Retorno ao normal ──────────────────────────────
        ("N",      860),
        ("N",      855),
        ("N",      858),
        ("N",      857),
        # ── FA simulada (batimentos 20–24, RR muito irregulares) ──
        ("FA",     420),
        ("FA",     810),
        ("FA",     380),
        ("FA",     920),
        ("FA",     350),
    ]

    r_positions: list[int] = []
    true_labels: list[str] = []

    cursor = int(0.5 * fs)  # começa em 0.5 s

    for beat_type, rr_ms in beats_plan:
        idx = cursor
        if idx >= total:
            break

        # Injeta forma de onda adequada
        if beat_type == "V":
            wave = _pvc_waveform(total, idx, amplitude=2.2, width_ms=160, fs=fs)
        elif beat_type in ("N", "TACHY", "FA"):
            wave = _gaussian_qrs(total, idx, amplitude=1.5, width_ms=45, fs=fs)
            wave += _t_wave(total, idx, amplitude=0.25, fs=fs)
        elif beat_type == "A":
            # APC: QRS ligeiramente mais estreito que normal
            wave = _gaussian_qrs(total, idx, amplitude=1.2, width_ms=38, fs=fs)
            wave += _t_wave(total, idx, amplitude=0.20, fs=fs)
        elif beat_type == "PAUSE":
            # Batimento normal após a pausa
            wave = _gaussian_qrs(total, idx, amplitude=1.5, width_ms=45, fs=fs)
            wave += _t_wave(total, idx, amplitude=0.25, fs=fs)
        else:
            wave = _gaussian_qrs(total, idx, amplitude=1.5, width_ms=45, fs=fs)

        signal += wave
        r_positions.append(idx)
        true_labels.append(beat_type)

        # Avança cursor pelo RR deste batimento
        cursor += int(rr_ms / 1000.0 * fs)

    return signal, np.array(r_positions, dtype=int), true_labels


# ──────────────────────────────────────────────────────────────
# 3. Execução do pipeline
# ──────────────────────────────────────────────────────────────

def run_demo():
    print("\n" + "═" * 60)
    print("  DEMO: ECG Sintético com Arritmias Forçadas")
    print("═" * 60)

    # ── Gera ECG ──────────────────────────────────────────────
    raw, r_true, true_labels = build_arrhythmia_ecg(FS)
    t = np.arange(len(raw)) / FS
    print(f"\n[demo] Sinal gerado: {len(raw)/FS:.1f} s | {len(r_true)} batimentos")
    print("[demo] Batimentos planejados:")
    for i, (pos, lbl) in enumerate(zip(r_true, true_labels)):
        print(f"       Beat {i+1:2d} @ {pos/FS:.2f}s → {lbl}")

    # ── Pan-Tompkins ──────────────────────────────────────────
    print("\n[demo] Rodando Pan-Tompkins …")
    stages = preprocess(raw, FS)
    r_peaks_mwi = adaptive_thresholding(stages["integrated"], FS)
    r_peaks = refine_r_peaks(r_peaks_mwi, stages["bandpass"], FS)
    print(f"[demo] Picos R detectados: {len(r_peaks)}")

    # ── Avaliação de detecção vs verdade ──────────────────────
    det_res = evaluate_detection(r_peaks, r_true, FS, tolerance_ms=75)
    detection_report(det_res, record_name="synthetic_arrhythmia")

    # ── Classificação ─────────────────────────────────────────
    clf = BeatClassifier()
    beat_labels, rhythm_info = clf.classify(stages["bandpass"], r_peaks, FS)
    hrv = compute_hrv_metrics(r_peaks, FS)

    print(f"[demo] Diagnóstico do Ritmo : {rhythm_info.get('rhythm', 'N/A')}")
    print(f"[demo] FC Média             : {rhythm_info.get('mean_bpm', 0):.1f} BPM")
    print(f"[demo] RMSSD                : {hrv.get('rmssd_ms', 0):.1f} ms")
    print(f"[demo] SDNN                 : {hrv.get('sdnn_ms', 0):.1f} ms")
    print(f"[demo] pNN50                : {hrv.get('pnn50', 0):.1f}%")

    # Contagem de arritmias detectadas
    from collections import Counter
    counts = Counter(beat_labels)
    print("\n[demo] Rótulos detectados pelo classificador:")
    for lbl, cnt in sorted(counts.items()):
        marker = "⚠" if lbl not in ("N",) else " "
        print(f"       {marker} {lbl:8s}: {cnt} batimento(s)")

    # ── Plot multi-painel ─────────────────────────────────────
    _plot_demo(t, raw, stages, r_peaks, r_true, true_labels,
               beat_labels, hrv, rhythm_info)

    print(f"\n[demo] Figura salva → ecg_synthetic_arrhythmia.png")
    print("═" * 60 + "\n")


# ──────────────────────────────────────────────────────────────
# 4. Plot personalizado para o demo
# ──────────────────────────────────────────────────────────────

def _plot_demo(t, raw, stages, r_peaks, r_true, true_labels,
               beat_labels, hrv, rhythm_info):
    """Figura 3 painéis focada na comparação verdade vs detectado."""

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), facecolor=PALETTE["bg"])

    rhythm_str = (
        f"{rhythm_info.get('rhythm', '')}  |  "
        f"{rhythm_info.get('mean_bpm', 0):.0f} BPM  |  "
        f"RMSSD={hrv.get('rmssd_ms', 0):.1f} ms"
    )
    fig.suptitle(
        f"ECG Sintético — Arritmias Forçadas\n{rhythm_str}",
        color=PALETTE["text"], fontsize=13, y=0.99,
    )

    # Cores por tipo de arritmia
    ARRHYTHMIA_COLOR = {
        "N":     "#50fa7b",
        "V":     "#ff5555",
        "A":     "#ffb86c",
        "TACHY": "#f1fa8c",
        "BRADY": "#8be9fd",
        "PAUSE": "#ff79c6",
        "FA":    "#bd93f9",
    }

    # ── Painel 1: ECG bruto + verdade (ground truth) ──────────
    ax1 = axes[0]
    ax1.set_facecolor(PALETTE["panel"])
    ax1.plot(t, raw, color=PALETTE["raw"], linewidth=0.8, alpha=0.9,
             label="ECG bruto")
    # Marca batimentos verdadeiros com triângulos
    for pos, lbl in zip(r_true, true_labels):
        if pos < len(raw):
            color = ARRHYTHMIA_COLOR.get(lbl, "#888")
            ax1.axvline(t[pos], color=color, alpha=0.5, linewidth=1.0,
                        linestyle="--")
            ax1.scatter(t[pos], raw[pos] + 0.3, marker="v",
                        color=color, s=60, zorder=5)
            ax1.text(t[pos], raw[pos] + 0.45, lbl,
                     color=color, fontsize=7, ha="center", fontweight="bold")

    ax1.set_title("1. ECG Bruto + Verdade (Ground Truth) — triângulos = tipo real",
                  color=PALETTE["text"], fontsize=10, pad=4)
    ax1.set_ylabel("mV", color=PALETTE["subtext"], fontsize=8)
    ax1.tick_params(colors=PALETTE["subtext"], labelsize=8)
    for spine in ax1.spines.values():
        spine.set_edgecolor(PALETTE["border"])

    # Legenda manual
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=c, label=f"{lbl}")
        for lbl, c in ARRHYTHMIA_COLOR.items()
        if lbl in set(true_labels)
    ]
    ax1.legend(handles=patches, facecolor=PALETTE["bg"],
               labelcolor=PALETTE["text"], fontsize=8,
               loc="upper right", ncol=len(patches))

    # ── Painel 2: ECG bandpass + picos R detectados coloridos ─
    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["panel"])
    bp = stages["bandpass"]
    ax2.plot(t, bp, color=PALETTE["bandpass"], linewidth=0.8, alpha=0.9)

    for idx, lbl in zip(r_peaks, beat_labels):
        if 0 <= idx < len(t):
            color = ARRHYTHMIA_COLOR.get(lbl, PALETTE["r_peak"])
            ax2.plot(t[idx], bp[idx], "o", color=color,
                     markersize=7, zorder=5, alpha=0.95)
            ax2.text(t[idx], bp[idx] + 0.25, lbl,
                     color=color, fontsize=7, ha="center", fontweight="bold")

    ax2.set_title(
        "2. Bandpass 5–15 Hz + Picos R Detectados — círculos = label classificado",
        color=PALETTE["text"], fontsize=10, pad=4,
    )
    ax2.set_ylabel("Amplitude", color=PALETTE["subtext"], fontsize=8)
    ax2.tick_params(colors=PALETTE["subtext"], labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor(PALETTE["border"])

    # ── Painel 3: RR Tachogram colorido por classe ────────────
    ax3 = axes[2]
    ax3.set_facecolor(PALETTE["panel"])

    if len(r_peaks) > 1:
        rr_ms = np.diff(r_peaks) / FS * 1000.0
        beat_n = np.arange(1, len(rr_ms) + 1)

        # Desenha segmentos coloridos por rótulo do batimento seguinte
        for i in range(len(rr_ms)):
            lbl = beat_labels[i + 1] if (i + 1) < len(beat_labels) else "N"
            color = ARRHYTHMIA_COLOR.get(lbl, PALETTE["rr_line"])
            ax3.plot(beat_n[i], rr_ms[i], "o", color=color,
                     markersize=8, zorder=5)
            if i > 0:
                prev_lbl = beat_labels[i] if i < len(beat_labels) else "N"
                ax3.plot([beat_n[i-1], beat_n[i]], [rr_ms[i-1], rr_ms[i]],
                         color=PALETTE["rr_line"], linewidth=1.0, alpha=0.5)

        rr_mean = float(np.mean(rr_ms))
        ax3.axhline(rr_mean, color=PALETTE["rr_mean"], linestyle="--",
                    linewidth=1.2,
                    label=f"Média {rr_mean:.0f} ms  "
                          f"SDNN={hrv.get('sdnn_ms',0):.1f} ms  "
                          f"RMSSD={hrv.get('rmssd_ms',0):.1f} ms")
        ax3.axhline(2000, color="#ff5555", linestyle=":", linewidth=1,
                    label="Limiar Pausa (2000 ms)")
        ax3.axhline(600, color="#ffb86c", linestyle=":", linewidth=1,
                    label="Limiar Taquicardia (600 ms = 100 BPM)")
        ax3.fill_between(beat_n, rr_ms, rr_mean,
                         alpha=0.10, color=PALETTE["rr_line"])
        ax3.legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["text"],
                   fontsize=8)

    ax3.set_title("3. RR Tachogram — pontos coloridos por classe detectada",
                  color=PALETTE["text"], fontsize=10, pad=4)
    ax3.set_xlabel("Número do batimento", color=PALETTE["subtext"], fontsize=8)
    ax3.set_ylabel("RR (ms)", color=PALETTE["subtext"], fontsize=8)
    ax3.tick_params(colors=PALETTE["subtext"], labelsize=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor(PALETTE["border"])

    for ax in axes[:2]:
        ax.set_xlabel("Tempo (s)", color=PALETTE["subtext"], fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTPUT_DIR / "ecg_synthetic_arrhythmia.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


if __name__ == "__main__":
    run_demo()