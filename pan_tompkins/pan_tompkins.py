"""
Pan-Tompkins Algorithm (PTA) — Detector de QRS
===============================================================================
Pipeline idêntico ao Verilog (filter.v → derivator.v → squarer.v →
integrator.v → peak_detector.v):

    1. Passa-banda  : filtros recursivos INTEIROS do Pan-Tompkins (LP+HP)
                      — mesmo que filter.v, sem float/Butterworth
    2. Derivada     : kernel [-1,-2,0,2,1] / 8, com sinal — mesmo que derivator.v
    3. Quadrado     : x² saturado em 32 bits — mesmo que squarer.v
    4. MWI          : média móvel de 150 ms (54 amostras @360 Hz) — integrator.v
    5. Detecção     : limiar = média móvel do MWI + refratário 200 ms
                      + learning phase de 2 s — peak_detector.v
    6. Relatório    : mesmo formato do testbench Verilog

NÃO usa scipy.butter nem find_peaks — aritmética inteira igual ao hardware.

Uso:
    python pan_tompkins.py                      # ECG sintético de demo
    python pan_tompkins.py --csv 100_signals.csv
    python pan_tompkins.py --wfdb 100 --dir .
    python pan_tompkins.py --mat ECG5.mat --fs 250 --var EKG5
    python pan_tompkins.py --csv sinal.csv --plot
"""

import argparse
import os
import sys

import numpy as np
import subprocess
import re


# ════════════════════════════════════════════════════════════════════
# CONSTANTES DO PIPELINE  (devem casar com os parâmetros Verilog)
# ════════════════════════════════════════════════════════════════════
BASELINE     = 128          # offset DC do sinal de 8 bits (0–255 → centrado em 0)
MWI_WINDOW   = 54           # amostras: 150 ms × 360 Hz  (WINDOW_SIZE do integrator.v)
REFRACTORY   = 72           # amostras: 200 ms × 360 Hz  (REFRACTORY do peak_detector.v)
LEARNING     = 720          # amostras:   2 s × 360 Hz   (LEARNING do peak_detector.v)
MAX_INT32    = 0xFFFFFFFF   # saturação do squarer.v


# ════════════════════════════════════════════════════════════════════
# ETAPA 1 — FILTRO PASSA-BANDA INTEIRO  (filter.v)
# ════════════════════════════════════════════════════════════════════
def etapa1_passabanda(ecg_raw):
    """
    Reproduz filter.v em Python:
      • Subtrai BASELINE (128) para centrar o sinal — elimina o DC
        que saturaria o integrador duplo do passa-baixa.
      • Passa-baixa  : y[n] = 2y[n-1] - y[n-2] + x[n] - 2x[n-6] + x[n-12]
      • Passa-alta   : y[n] = 32x[n-16] - (y[n-1] + x[n] - x[n-32])
      Coeficientes inteiros originais de Pan & Tompkins (1985).
    """
    n = len(ecg_raw)

    # ── Passa-baixa ──
    lp = np.zeros(n, dtype=np.int64)
    x  = np.zeros(13, dtype=np.int64)
    y1, y2 = np.int64(0), np.int64(0)
    for i in range(n):
        # desloca histórico
        x[12:0:-1] = x[11::-1]
        x[0] = int(ecg_raw[i]) - BASELINE          # remove DC, igual ao filter.v
        val  = np.int64(2)*y1 - y2 + x[0] - np.int64(2)*x[6] + x[12]
        y2, y1 = y1, val
        lp[i] = val

    # ── Passa-alta ──
    hp  = np.zeros(n, dtype=np.int64)
    hx  = np.zeros(33, dtype=np.int64)
    hy1 = np.int64(0)
    for i in range(n):
        hx[32:0:-1] = hx[31::-1]
        hx[0] = lp[i]
        val   = np.int64(32)*hx[16] - (hy1 + lp[i] - hx[32])
        hy1   = val
        hp[i] = val

    return hp


# ════════════════════════════════════════════════════════════════════
# ETAPA 2 — DERIVADA DE 5 PONTOS  (derivator.v)
# ════════════════════════════════════════════════════════════════════
def etapa2_derivada(bp):
    """
    Reproduz derivator.v: (2x[0]+x[1]-x[3]-2x[4]) >> 3, COM sinal.
    """
    n = len(bp)
    out = np.zeros(n, dtype=np.int64)
    buf = np.zeros(5, dtype=np.int64)
    for i in range(n):
        buf[4], buf[3], buf[2], buf[1] = buf[3], buf[2], buf[1], buf[0]
        buf[0] = bp[i]
        d = np.int64(2)*buf[0] + buf[1] - buf[3] - np.int64(2)*buf[4]
        out[i] = d >> 3          # normaliza /8 preservando sinal (>>3 aritmético)
    return out


# ════════════════════════════════════════════════════════════════════
# ETAPA 3 — QUADRADO  (squarer.v)
# ════════════════════════════════════════════════════════════════════
def etapa3_quadrado(der):
    """
    Reproduz squarer.v: x² saturado em 32 bits (uint32).
    """
    sq = der.astype(np.int64) ** 2
    sq = np.clip(sq, 0, MAX_INT32)
    return sq.astype(np.uint64)


# ════════════════════════════════════════════════════════════════════
# ETAPA 4 — MWI — MÉDIA MÓVEL  (integrator.v)
# ════════════════════════════════════════════════════════════════════
def etapa4_mwi(sq, window=MWI_WINDOW):
    """
    Reproduz integrator.v: soma corrente dividida pelo WINDOW_SIZE,
    com divisão inteira — igual a data_out = sum / WINDOW_SIZE.
    """
    n = len(sq)
    out  = np.zeros(n, dtype=np.uint64)
    buf  = np.zeros(window, dtype=np.uint64)
    soma = np.uint64(0)
    ptr  = 0
    for i in range(n):
        soma = soma - buf[ptr] + sq[i]
        buf[ptr] = sq[i]
        ptr = (ptr + 1) % window
        out[i] = soma // window
    return out


# ════════════════════════════════════════════════════════════════════
# ETAPA 5 — DETECÇÃO DE PICOS  (peak_detector.v)
# ════════════════════════════════════════════════════════════════════
def etapa5_detecta_picos(mwi, fs,
                          refractory=REFRACTORY,
                          learning=LEARNING):
    """
    Reproduz peak_detector.v — threshold adaptativo DUPLO SPKI/NPKI
    (Pan-Tompkins clássico), idêntico ao Verilog:

      Fase de aprendizado (LEARNING amostras = 2 s):
        SPKI ← média do MWI nos primeiros 2 s
        NPKI ← SPKI / 2  (estimativa conservadora do ruído de fundo)
        THR  ← NPKI + (SPKI - NPKI) / 4  = NPKI*0.75 + SPKI*0.25

      Detecção normal (após aprendizado, fora do refratário):
        Pico aceito  (prev > curr AND prev > THR):
            SPKI ← 0.875*SPKI + 0.125*amp   (>> 3 em hardware)
            THR  ← 3/4*NPKI + 1/4*SPKI_novo
        Candidato rejeitado (abaixo do THR):
            NPKI ← 0.875*NPKI + 0.125*amp
            THR  ← 3/4*NPKI_novo + 1/4*SPKI

      Refratário: bloqueia REFRACTORY amostras após cada pico aceito.
    """
    n = len(mwi)
    picos = []

    SPKI = np.uint64(1)
    NPKI = np.uint64(0)
    THR  = np.uint64(0)

    prev      = np.uint64(0)
    refr_cnt  = 0
    learn_cnt = 0
    learn_sum = np.uint64(0)
    subindo   = False

    for i in range(n):
        x = np.uint64(mwi[i])

        # ── Fase de aprendizado ──
        if learn_cnt < learning:
            learn_sum += x
            learn_cnt += 1
            if learn_cnt == learning:
                # mesmo cálculo do Verilog:
                # SPKI = (learn_sum + LEARNING) / LEARNING  (arredondamento)
                SPKI = (learn_sum + np.uint64(learning)) // np.uint64(learning)
                NPKI = SPKI >> np.uint64(1)
                THR  = NPKI - (NPKI >> np.uint64(2)) + (SPKI >> np.uint64(2))

        # ── Período refratário ──
        elif refr_cnt > 0:
            refr_cnt -= 1

        # ── Detecção normal ──
        else:
            if x > THR:
                subindo = True

            if subindo and prev > x and prev > THR:
                # Pico aceito — atualiza SPKI
                SPKI_novo = SPKI - (SPKI >> np.uint64(3)) + (prev >> np.uint64(3))
                THR = NPKI - (NPKI >> np.uint64(2)) + (SPKI_novo >> np.uint64(2))
                SPKI = SPKI_novo

                picos.append(i - 1)
                refr_cnt = refractory
                subindo  = False

            elif not subindo and x <= THR:
                # Candidato rejeitado — atualiza NPKI
                NPKI_novo = NPKI - (NPKI >> np.uint64(3)) + (x >> np.uint64(3))
                THR = NPKI_novo - (NPKI_novo >> np.uint64(2)) + (SPKI >> np.uint64(2))
                NPKI = NPKI_novo

        prev = x

    return np.array(picos, dtype=int)


# ════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO
# ════════════════════════════════════════════════════════════════════
def pan_tompkins(ecg_raw, fs, retornar_etapas=False):
    """
    Executa as 5 etapas idênticas ao Verilog e devolve os picos R.
    Nota: sem etapa 6 (refinamento) — o Verilog também não a tem no HW.
    """
    ecg_raw = np.asarray(ecg_raw).squeeze()

    bp  = etapa1_passabanda(ecg_raw)
    der = etapa2_derivada(bp)
    sq  = etapa3_quadrado(der)
    mwi = etapa4_mwi(sq)
    r_peaks = etapa5_detecta_picos(mwi, fs)

    if retornar_etapas:
        return r_peaks, {"bruto": ecg_raw, "passabanda": bp,
                         "derivada": der, "quadrado": sq, "mwi": mwi}
    return r_peaks


# ════════════════════════════════════════════════════════════════════
# MEDIDAS DERIVADAS
# ════════════════════════════════════════════════════════════════════
def medidas_dos_picos(r_peaks, fs):
    res = dict(n_picos=len(r_peaks), tempos_s=np.array([]),
               rr_ms=np.array([]), fc_inst=np.array([]),
               fc_media=0.0, fc_min=0.0, fc_max=0.0,
               rr_medio_ms=0.0, rr_min_ms=0.0, rr_max_ms=0.0,
               sdnn_ms=0.0, rmssd_ms=0.0, duracao_s=0.0)
    if len(r_peaks) == 0:
        return res

    tempos = r_peaks / fs
    res["tempos_s"] = tempos
    res["duracao_s"] = float(tempos[-1] - tempos[0]) if len(tempos) > 1 else 0.0

    if len(r_peaks) < 2:
        return res

    rr_ms = np.diff(r_peaks) / fs * 1000.0
    fc    = 60000.0 / rr_ms

    res["rr_ms"]      = rr_ms
    res["fc_inst"]    = fc
    res["fc_media"]   = float(np.mean(fc))
    res["fc_min"]     = float(np.min(fc))
    res["fc_max"]     = float(np.max(fc))
    res["rr_medio_ms"]= float(np.mean(rr_ms))
    res["rr_min_ms"]  = float(np.min(rr_ms))
    res["rr_max_ms"]  = float(np.max(rr_ms))
    res["sdnn_ms"]    = float(np.std(rr_ms))
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms)**2))) if len(rr_ms) > 1 else 0.0
    res["rmssd_ms"]   = rmssd
    return res


# ════════════════════════════════════════════════════════════════════
# RELATÓRIO — mesmo formato do testbench Verilog
# ════════════════════════════════════════════════════════════════════
def imprimir_relatorio(nome, fs, etapas, m, max_listar=12):
    L = 64

    print("\n" + "=" * L)
    print(f"  ECG PROCESSOR - PAN-TOMPKINS (Python ≡ Verilog)")
    print(f"  Registro: {nome}")
    print("=" * L)
    print(f"  Frequencia de amostragem (fs) : {fs} Hz")
    n = len(etapas["bruto"])
    print(f"  Amostras processadas          : {n:,}  ({n//fs} s)")

    # ── Etapas ──
    print("-" * L)
    print("  ETAPAS DO PIPELINE  (modulo Verilog -> faixa de amplitude)")
    print("-" * L)
    def faixa(x, label):
        arr = np.asarray(x, dtype=np.int64)
        print(f"  {label:<30}  [{np.min(arr):>12,} .. {np.max(arr):>+12,}]")
    faixa(etapas["passabanda"], "filter.v      (passa-banda)")
    faixa(etapas["derivada"],   "derivator.v   (/8 c/ sinal)")
    faixa(etapas["quadrado"],   "squarer.v     (x²)")
    faixa(etapas["mwi"],        "integrator.v  (MWI)")

    # ── Picos R ──
    print("-" * L)
    print("  PICOS R DETECTADOS  (peak_detector.v -> alert_out)")
    print("-" * L)
    print(f"  Total de picos R : {m['n_picos']}")

    if m["n_picos"] == 0:
        print("  (nenhum pico detectado)")
        print("=" * L + "\n")
        return

    tempos = m["tempos_s"]
    listar = min(max_listar, len(tempos))
    print(f"  {'#':>4}  {'Amostra':>9}  {'Tempo':>10}")
    print(f"  {'────':>4}  {'────────':>9}  {'──────────':>10}")
    for i in range(listar):
        am = int(round(tempos[i] * fs))
        print(f"  R{i+1:>3}  {am:>9}  {tempos[i]:>9.3f} s")
    if len(tempos) > listar:
        print(f"   ... (+{len(tempos) - listar} picos)")

    # ── Intervalos RR ──
    if len(m["rr_ms"]) > 0:
        rr = m["rr_ms"]
        print("-" * L)
        print("  INTERVALOS RR  (total: %d)" % len(rr))
        print("-" * L)
        print(f"  RR medio  : {m['rr_medio_ms']:8.1f} ms")
        print(f"  RR minimo : {m['rr_min_ms']:8.1f} ms")
        print(f"  RR maximo : {m['rr_max_ms']:8.1f} ms")
        print("  " + "·" * (L - 2))
        listar_rr = min(max_listar, len(rr))
        for i in range(listar_rr):
            fc_i = 60000.0 / rr[i]
            print(f"    RR{i+1:<4}: {rr[i]:8.1f} ms   (FC = {fc_i:5.1f} bpm)")
        if len(rr) > listar_rr:
            print(f"    ... (+{len(rr) - listar_rr} intervalos)")

    # ── FC ──
    print("-" * L)
    print("  FREQUENCIA CARDIACA  (FC / BPM)")
    print("-" * L)
    print(f"  FC media  : {m['fc_media']:6.1f} bpm")
    print(f"  FC minima : {m['fc_min']:6.1f} bpm")
    print(f"  FC maxima : {m['fc_max']:6.1f} bpm")

    # ── Variabilidade ──
    print("-" * L)
    print("  VARIABILIDADE DO RR  (descritivo - sem diagnostico clinico)")
    print("-" * L)
    print(f"  SDNN   (desvio padrao dos RR)          : {m['sdnn_ms']:6.1f} ms")
    print(f"  RMSSD  (dif. quadratica media sucessiva): {m['rmssd_ms']:6.1f} ms")
    print(f"  Janela analisada                        : {m['duracao_s']:.1f} s")
    print("=" * L)
    print("  Este projeto SO detecta QRS e mede RR/FC.")
    print("=" * L + "\n")


# ════════════════════════════════════════════════════════════════════
# CARREGAMENTO DE SINAL
# ════════════════════════════════════════════════════════════════════
def carregar_csv(path, max_amostras=65000):
    """
    Carrega CSV com coluna time_ms + coluna de sinal.
    - Limita a max_amostras (padrão 65.000, igual ao testbench Verilog).
    - Normaliza para 0-255 com min-max global do segmento carregado.
    """
    import pandas as pd
    df = pd.read_csv(path)
    if "time_ms" in df.columns:
        col = [c for c in df.columns if c != "time_ms"][0]
        t   = df["time_ms"].values
        fs  = int(round(1000.0 / np.median(np.diff(t)))) if len(t) > 1 else None
    else:
        col = df.columns[0]
        fs  = None

    sig = df[col].values.astype(float)

    if len(sig) > max_amostras:
        sig = sig[:max_amostras]

    if sig.min() < 0 or sig.max() > 255:
        sig = (sig - sig.min()) / (sig.max() - sig.min()) * 255

    return sig.astype(np.uint8), fs, col


def carregar_mat(path, var=None):
    from scipy.io import loadmat
    d = loadmat(path)
    chaves = [k for k in d.keys() if not k.startswith("__")]
    chave  = var if (var and var in d) else chaves[0]
    return np.asarray(d[chave]).squeeze(), chave


def carregar_wfdb(record, data_dir="."):
    try:
        import wfdb
    except ImportError:
        sys.exit("pip install wfdb")
    rec  = wfdb.rdrecord(os.path.join(data_dir, record))
    fs   = int(rec.fs)
    idx  = rec.sig_name.index("MLII") if "MLII" in rec.sig_name else 0
    canal= rec.sig_name[idx] if rec.sig_name else "ch0"
    # converte mV → 8 bits para casar com o pipeline inteiro (igual ao .mem)
    sig  = rec.p_signal[:, idx]
    sig  = (sig - sig.min()) / (sig.max() - sig.min()) * 255
    return sig.astype(np.uint8), fs, canal


def ecg_sintetico(fs=360, dur_s=20, bpm=72, seed=1):
    """ECG sintético de 8 bits, igual ao gerado para o testbench Verilog."""
    np.random.seed(seed)
    n   = int(fs * dur_s)
    sig = np.zeros(n)
    rr  = int(round(60.0 / bpm * fs))
    larg = int(0.02 * fs)
    for r in range(rr, n, rr):
        for j in range(-larg, larg):
            if 0 <= r+j < n:
                sig[r+j] += (larg - abs(j)) / larg * 60
    for r in range(rr, n, rr):
        tp = r + int(0.20 * fs); wT = int(0.06 * fs)
        for j in range(-wT, wT):
            if 0 <= tp+j < n:
                sig[tp+j] += (wT - abs(j)) / wT * 15
    sig += np.random.randn(n) * 2
    sig  = np.clip(sig + 128, 0, 255).astype(np.uint8)
    return sig, fs


def salvar_figura(etapas, r_peaks, fs,
                  nome="pan_tompkins_etapas.png", seg_s=6):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [aviso] pip install matplotlib para gerar figura")
        return None
    n = min(int(seg_s * fs), len(etapas["bruto"]))
    t = np.arange(n) / fs
    fig, ax = plt.subplots(5, 1, figsize=(11, 9), sharex=True)
    labels = ["Bruto (8-bit)", "Passa-banda (inteiro)", "Derivada²", "MWI", "Picos R"]
    cores  = ["#888888", "#4a90d9", "#f0a500", "#9b72cf", "#888888"]
    for k, (a, lbl, cor) in enumerate(zip(ax, labels, cores)):
        key = ["bruto","passabanda","quadrado","mwi","bruto"][k]
        a.plot(t, np.asarray(etapas[key][:n], dtype=float), color=cor, lw=0.8)
        a.set_ylabel(lbl, fontsize=8)
        a.grid(True, alpha=0.2)
    rp = r_peaks[r_peaks < n]
    ax[4].plot(rp/fs, np.asarray(etapas["bruto"])[rp], "rv", ms=7, label="R")
    ax[4].legend(fontsize=8)
    ax[4].set_xlabel("Tempo (s)")
    fig.suptitle("Pan-Tompkins — pipeline inteiro (Python ≡ Verilog)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(nome, dpi=110)
    plt.close(fig)
    return nome


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="Pan-Tompkins — pipeline inteiro idêntico ao Verilog."
    )
    p.add_argument("--csv",   help="CSV (time_ms + sinal)")
    p.add_argument("--mat",   help="Arquivo .mat")
    p.add_argument("--var",   help="Variável dentro do .mat")
    p.add_argument("--wfdb",  help="Registro WFDB (ex.: 100)")
    p.add_argument("--dir",   default=".", help="Pasta dos arquivos WFDB")
    p.add_argument("--fs",    type=int, help="fs em Hz (se não inferido)")
    p.add_argument("--plot",  action="store_true", help="Salva figura das etapas")
    p.add_argument("--listar",   type=int, default=12, help="Picos a listar")
    p.add_argument("--amostras", type=int, default=65000,
                   help="Max amostras a processar (padrão 65.000 = igual ao Verilog)")
    args = p.parse_args()

    def run_download_csv(arg):
        # call download_csv.py in repo root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        script = os.path.join(repo_root, "download_csv.py")
        if not os.path.exists(script):
            sys.exit("download_csv.py not found in repository root")

        # prepare command depending on arg
        if arg == "all":
            cmd = ["python3", script, "all"]
            out_name = os.path.join(repo_root, "all_signals_combined.csv")
        elif arg.isdigit():
            cmd = ["python3", script, arg]
            out_name = os.path.join(repo_root, f"{arg}_signals.csv")
        else:
            # treat as file id or url
            cmd = ["python3", script, "file", "--file-id", arg]
            out_name = os.path.join(repo_root, "file_signals.csv")

        try:
            subprocess.run(cmd, check=True, cwd=repo_root)
        except subprocess.CalledProcessError as e:
            sys.exit(f"download_csv.py failed: {e}")
        return out_name

    if args.csv:
        # if args.csv is an existing local path, use it directly
        if os.path.exists(args.csv):
            sig, fs_inf, nome = carregar_csv(args.csv, max_amostras=args.amostras)
            fs = args.fs or fs_inf
            nome = f"{os.path.basename(args.csv)} ({nome})"
        else:
            # if args.csv looks like a number, 'all', or drive id/link, download it
            is_drive_link = bool(re.search(r"drive\.google\.com|/d/", str(args.csv)))
            is_id_like = bool(re.fullmatch(r"[A-Za-z0-9_-]{10,}", str(args.csv)))
            if args.csv == "all" or args.csv.isdigit() or is_drive_link or is_id_like:
                downloaded = run_download_csv(args.csv)
                if not os.path.exists(downloaded):
                    sys.exit(f"Downloaded file not found: {downloaded}")
                sig, fs_inf, nome = carregar_csv(downloaded, max_amostras=args.amostras)
                fs = args.fs or fs_inf
                nome = f"{os.path.basename(downloaded)} ({nome})"
            else:
                # treat as path even if it doesn't exist locally
                sig, fs_inf, nome = carregar_csv(args.csv, max_amostras=args.amostras)
                fs = args.fs or fs_inf
                nome = f"{os.path.basename(args.csv)} ({nome})"
    elif args.mat:
        sig, var = carregar_mat(args.mat, args.var)
        fs = args.fs
        nome = f"{os.path.basename(args.mat)} ({var})"
        if fs is None:
            sys.exit("Informe --fs para arquivos .mat.")
    elif args.wfdb:
        sig, fs, canal = carregar_wfdb(args.wfdb, args.dir)
        nome = f"WFDB {args.wfdb} ({canal})"
    else:
        sig, fs = ecg_sintetico()
        nome = "ECG sintetico (demo 72 bpm, 360 Hz, 20 s)"
        print("\n[info] Sem arquivo — usando ECG sintetico de demo.")

    if fs is None:
        sys.exit("Nao foi possivel determinar fs. Use --fs.")

    r_peaks, etapas = pan_tompkins(sig, fs, retornar_etapas=True)
    m = medidas_dos_picos(r_peaks, fs)
    imprimir_relatorio(nome, fs, etapas, m, max_listar=args.listar)

    if args.plot:
        out = salvar_figura(etapas, r_peaks, fs)
        if out:
            print(f"  Figura salva em: {out}\n")


if __name__ == "__main__":
    main()