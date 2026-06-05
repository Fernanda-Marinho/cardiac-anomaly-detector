"""
metrics.py – Avaliação de desempenho da detecção e classificação
================================================================
Funções:
  evaluate_detection  – TP/FP/FN com tolerância temporal (padrão 50 ms)
  detection_report    – imprime tabela formatada de métricas de detecção
  classification_report_beats – métricas por classe de batimento
  compare_records     – avalia o pipeline em múltiplos registros MIT-BIH
"""

from __future__ import annotations

from typing import Optional
import numpy as np


# ──────────────────────────────────────────────────────────────
# Avaliação de detecção de picos R
# ──────────────────────────────────────────────────────────────

def evaluate_detection(
    detected: np.ndarray,
    reference: np.ndarray,
    fs: int,
    tolerance_ms: float = 50.0,
) -> dict:
    """
    Calcula métricas de detecção com janela de tolerância.

    Um pico detectado é considerado verdadeiro positivo (TP) se existe
    pelo menos um pico de referência dentro da tolerância.
    Cada pico de referência pode casar com no máximo um pico detectado.

    Parâmetros
    ----------
    detected     : posições dos picos R detectados (amostras)
    reference    : posições dos picos R de referência (amostras)
    fs           : frequência de amostragem
    tolerance_ms : janela de tolerância em ms (padrão AAMI = 50 ms)

    Retorna
    -------
    dict com: tp, fp, fn, sensitivity, ppv, f1, tolerance_ms
    """
    tol = int(tolerance_ms / 1000.0 * fs)
    detected = np.sort(detected)
    reference = np.sort(reference)

    tp = 0
    ref_matched = np.zeros(len(reference), dtype=bool)

    for d_idx in detected:
        diffs = np.abs(reference - d_idx)
        candidates = np.where(diffs <= tol)[0]
        # Pega o candidato mais próximo ainda não casado
        for c in candidates[np.argsort(diffs[candidates])]:
            if not ref_matched[c]:
                tp += 1
                ref_matched[c] = True
                break

    fp = len(detected) - tp
    fn = len(reference) - tp

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * sensitivity * ppv / (sensitivity + ppv)
          if (sensitivity + ppv) > 0 else 0.0)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "ppv": ppv,
        "f1": f1,
        "tolerance_ms": tolerance_ms,
        "n_detected": len(detected),
        "n_reference": len(reference),
    }


def detection_report(
    results: dict,
    record_name: str = "",
    verbose: bool = True,
) -> str:
    """
    Formata e (opcionalmente) imprime o relatório de detecção.

    Retorna str com o relatório formatado.
    """
    hdr = f"  Record {record_name}" if record_name else "  Resultados"
    lines = [
        "",
        "═" * 52,
        f"{hdr} — Avaliação da Detecção de QRS",
        "─" * 52,
        f"  Tolerância          : {results['tolerance_ms']:.0f} ms",
        f"  Picos detectados    : {results['n_detected']}",
        f"  Referência (anotações): {results['n_reference']}",
        "─" * 52,
        f"  Verdadeiros Positivos (TP): {results['tp']}",
        f"  Falsos  Positivos   (FP) : {results['fp']}",
        f"  Falsos  Negativos   (FN) : {results['fn']}",
        "─" * 52,
        f"  Sensibilidade       : {results['sensitivity'] * 100:.2f} %",
        f"  Val. Pred. Positiva : {results['ppv'] * 100:.2f} %",
        f"  F1-Score            : {results['f1'] * 100:.2f} %",
        "═" * 52,
        "",
    ]
    report = "\n".join(lines)
    if verbose:
        print(report)
    return report


# ──────────────────────────────────────────────────────────────
# Avaliação da classificação beat-a-beat
# ──────────────────────────────────────────────────────────────

def classification_report_beats(
    y_true: list[str],
    y_pred: list[str],
    labels: Optional[list[str]] = None,
    verbose: bool = True,
) -> dict:
    """
    Métricas de classificação por classe de batimento.

    Calcula precision, recall e F1 por classe sem depender de sklearn,
    mas usa sklearn se disponível para saída mais rica.

    Parâmetros
    ----------
    y_true  : rótulos verdadeiros (símbolos AAMI, ex: ["N","N","V","A"])
    y_pred  : rótulos preditos
    labels  : lista de classes a incluir (None = todas encontradas)

    Retorna
    -------
    dict com 'per_class' (dict por classe) e 'macro_f1'
    """
    y_true = list(y_true)
    y_pred = list(y_pred)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    per_class = {}
    for lbl in labels:
        tp = sum(t == lbl and p == lbl for t, p in zip(y_true, y_pred))
        fp = sum(t != lbl and p == lbl for t, p in zip(y_true, y_pred))
        fn = sum(t == lbl and p != lbl for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[lbl] = {"tp": tp, "fp": fp, "fn": fn,
                          "precision": prec, "recall": rec, "f1": f1}

    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))

    if verbose:
        print("\n" + "═" * 56)
        print("  Classificação Beat-a-Beat")
        print("─" * 56)
        print(f"  {'Classe':<10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'TP':>6}")
        print("─" * 56)
        for lbl, m in sorted(per_class.items()):
            print(
                f"  {lbl:<10} {m['precision']*100:>9.1f}%"
                f" {m['recall']*100:>9.1f}%"
                f" {m['f1']*100:>7.1f}%"
                f" {m['tp']:>6}"
            )
        print("─" * 56)
        print(f"  Macro F1: {macro_f1 * 100:.2f} %")
        print("═" * 56 + "\n")

    return {"per_class": per_class, "macro_f1": macro_f1}


# ──────────────────────────────────────────────────────────────
# Avaliação em múltiplos registros
# ──────────────────────────────────────────────────────────────

def compare_records(
    records: list[str],
    pipeline_fn,          # callable(record_name) → (r_peaks, reference, fs)
    tolerance_ms: float = 50.0,
    verbose: bool = True,
) -> dict:
    """
    Avalia o pipeline em múltiplos registros e agrega métricas.

    Parâmetros
    ----------
    records     : lista de IDs de registro MIT-BIH
    pipeline_fn : função que recebe record_name e retorna
                  (r_peaks_detected, r_peaks_reference, fs)
    tolerance_ms: tolerância de detecção

    Retorna
    -------
    dict com métricas agregadas (média ± desvio) e por registro
    """
    all_tp, all_fp, all_fn = 0, 0, 0
    per_record = {}

    for rec in records:
        try:
            detected, reference, fs = pipeline_fn(rec)
            res = evaluate_detection(detected, reference, fs, tolerance_ms)
            per_record[rec] = res
            all_tp += res["tp"]
            all_fp += res["fp"]
            all_fn += res["fn"]
            if verbose:
                print(f"  {rec}: Se={res['sensitivity']*100:.1f}% "
                      f"PPV={res['ppv']*100:.1f}% "
                      f"F1={res['f1']*100:.1f}%")
        except Exception as e:
            print(f"  [ERRO] {rec}: {e}")

    global_se = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    global_ppv = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    global_f1 = (2 * global_se * global_ppv / (global_se + global_ppv)
                 if (global_se + global_ppv) > 0 else 0.0)

    summary = {
        "global_sensitivity": global_se,
        "global_ppv": global_ppv,
        "global_f1": global_f1,
        "total_tp": all_tp,
        "total_fp": all_fp,
        "total_fn": all_fn,
        "per_record": per_record,
    }

    if verbose:
        print(f"\n  GLOBAL → Se={global_se*100:.2f}%  "
              f"PPV={global_ppv*100:.2f}%  F1={global_f1*100:.2f}%")

    return summary
