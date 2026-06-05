"""
main.py – Ponto de entrada do detector de anomalias cardíacas
=============================================================
Uso:
    python main.py                          # demo: record 100, 30 s
    python main.py --record 208 --duration 60
    python main.py --record 100 --train     # treina RF com anotações
    python main.py --records 100 200 208    # avalia múltiplos registros
    python main.py --list                   # lista registros disponíveis
    python main.py --help

Saídas geradas:
    ecg_<record>.png            – figura 7 painéis
    confusion_matrix.png        – (se --train)
    data/processed/<record>.csv – série temporal exportada
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Adiciona src/ ao path para facilitar execução direta
sys.path.insert(0, str(Path(__file__).parent))

from src.loader import load_record, list_records, export_csv, AAMI_MAP
from src.pan_tompkins import pan_tompkins
from src.features import extract_beat_features, compute_hrv_metrics
from src.classifier import BeatClassifier
from src.plotter import plot_results, plot_confusion_matrix
from src.metrics import (
    evaluate_detection,
    detection_report,
    classification_report_beats,
    compare_records,
)

DATA_DIR = Path(__file__).parent / "data"


# ──────────────────────────────────────────────────────────────
# Funções auxiliares
# ──────────────────────────────────────────────────────────────

def run_pipeline(
    record_name: str,
    duration_sec: int = 30,
    train: bool = False,
    export: bool = True,
    output_dir: Path = Path("."),
) -> dict:
    """
    Executa o pipeline completo em um registro.

    Retorna dict com todos os resultados para uso externo.
    """
    t0 = time.time()

    # ── 1. Carrega dado ───────────────────────────────────────
    raw, fs, ann, t = load_record(
        record_name=record_name,
        duration_sec=duration_sec,
        data_dir=DATA_DIR,
    )

    # ── 2. Pan-Tompkins ───────────────────────────────────────
    print(f"\n[main] Executando Pan-Tompkins em record {record_name} …")
    stages = pan_tompkins(raw, fs)
    r_peaks = stages["r_peaks"]
    print(f"[main] {len(r_peaks)} picos R detectados")

    # ── 3. HRV e métricas de detecção ────────────────────────
    hrv = compute_hrv_metrics(r_peaks, fs)

    # Referência: todos os batimentos anotados
    ref_all = ann.sample
    detection_res = evaluate_detection(ref_all, ref_all, fs)  # baseline
    # Avaliação real: detectados vs referência
    detection_res = evaluate_detection(r_peaks, ref_all, fs)
    detection_report(detection_res, record_name=record_name)

    # ── 4. Classificação de batimentos ────────────────────────
    clf = BeatClassifier()

    if train:
        print("[main] Treinando classificador ML com anotações do registro …")
        # Alinhar r_peaks com anotações para treino
        ann_symbols = _align_annotations(r_peaks, ann, fs)
        clf.train(stages["bandpass"], r_peaks, ann_symbols, fs)

    beat_labels, rhythm_info = clf.classify(stages["bandpass"], r_peaks, fs)

    print(f"\n[main] Diagnóstico do Ritmo: {rhythm_info.get('rhythm', 'N/A')}")
    print(f"[main] FC Média: {rhythm_info.get('mean_bpm', 0):.1f} BPM")
    print(f"[main] RMSSD: {hrv.get('rmssd_ms', 0):.1f} ms | "
          f"SDNN: {hrv.get('sdnn_ms', 0):.1f} ms | "
          f"pNN50: {hrv.get('pnn50', 0):.1f}%")

    # Sumário de arritmias detectadas
    arrhythmia_counts = {}
    for lbl in beat_labels:
        if lbl != "N":
            arrhythmia_counts[lbl] = arrhythmia_counts.get(lbl, 0) + 1

    if arrhythmia_counts:
        print("[main] Arritmias detectadas:")
        for lbl, cnt in sorted(arrhythmia_counts.items()):
            from src.loader import AAMI_MAP
            name = AAMI_MAP.get(lbl, lbl)
            print(f"       {lbl} ({name}): {cnt} batimento(s)")
    else:
        print("[main] Nenhuma arritmia detectada — ritmo aparentemente normal.")

    # ── 5. Plot ───────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"ecg_{record_name}.png"
    plot_results(t, raw, stages, r_peaks, ann, hrv,
                 beat_labels=beat_labels,
                 rhythm_info=rhythm_info,
                 output_path=fig_path)

    # ── 6. Exporta CSV ────────────────────────────────────────
    if export:
        csv_path = DATA_DIR / "processed" / f"{record_name}.csv"
        export_csv(raw, t, r_peaks, beat_labels, csv_path)

    elapsed = time.time() - t0
    print(f"\n[main] Concluído em {elapsed:.2f} s")

    return {
        "record": record_name,
        "r_peaks": r_peaks,
        "beat_labels": beat_labels,
        "rhythm_info": rhythm_info,
        "hrv": hrv,
        "detection": detection_res,
        "stages": stages,
    }


def _align_annotations(
    r_peaks: np.ndarray,
    ann,
    fs: int,
    tol_ms: float = 75.0,
) -> list[str]:
    """
    Alinha picos R detectados com anotações MIT-BIH para obter rótulos.

    Para cada pico R detectado, busca a anotação mais próxima dentro da
    tolerância. Se não encontrar, rotula como "N".
    """
    tol = int(tol_ms / 1000.0 * fs)
    labels = []

    for idx in r_peaks:
        diffs = np.abs(ann.sample - idx)
        best = int(np.argmin(diffs))
        if diffs[best] <= tol:
            labels.append(ann.symbol[best])
        else:
            labels.append("N")

    return labels


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detector de Anomalias Cardíacas — MIT-BIH / Pan-Tompkins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python main.py                         # demo rápido (record 100, 30 s)
  python main.py --record 200 --duration 60
  python main.py --record 208 --train    # treina classificador ML
  python main.py --records 100 200 208   # avalia múltiplos registros
  python main.py --list                  # lista todos os registros MIT-BIH
        """,
    )
    p.add_argument("--record",   default="100",
                   help="ID do registro MIT-BIH (padrão: 100)")
    p.add_argument("--duration", type=int, default=30,
                   help="Duração em segundos a analisar (padrão: 30)")
    p.add_argument("--train",    action="store_true",
                   help="Treina o classificador ML com anotações do registro")
    p.add_argument("--records",  nargs="+",
                   help="Avalia múltiplos registros e exibe métricas agregadas")
    p.add_argument("--list",     action="store_true",
                   help="Lista todos os registros MIT-BIH disponíveis")
    p.add_argument("--output",   default=".",
                   help="Diretório de saída para figuras (padrão: .)")
    p.add_argument("--no-export", action="store_true",
                   help="Não exporta CSV dos resultados")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output)

    # ── Listar registros ──────────────────────────────────────
    if args.list:
        recs = list_records()
        print("\nRegistros MIT-BIH disponíveis:")
        for i, r in enumerate(recs):
            end = "\n" if (i + 1) % 8 == 0 else "  "
            print(f"  {r}", end=end)
        print("\n")
        return

    # ── Múltiplos registros ───────────────────────────────────
    if args.records:
        print(f"\n[main] Avaliando {len(args.records)} registros …\n")

        def pipeline_fn(rec: str):
            raw, fs, ann, t = load_record(
                record_name=rec,
                duration_sec=args.duration,
                data_dir=DATA_DIR,
            )
            stages = pan_tompkins(raw, fs)
            return stages["r_peaks"], ann.sample, fs

        compare_records(args.records, pipeline_fn, tolerance_ms=50)
        return

    # ── Registro único ────────────────────────────────────────
    run_pipeline(
        record_name=args.record,
        duration_sec=args.duration,
        train=args.train,
        export=not args.no_export,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
