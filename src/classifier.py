"""
classifier.py – Classificador de batimentos / arritmias
=========================================================
Dois estágios de classificação:

Estágio 1 – Regras clínicas (determinístico):
  • Taquicardia / Bradicardia (frequência)
  • Flutter/Fibrilação Atrial (irregularidade do RR)
  • PVC (extrassístole ventricular) por RR prematuro + morfologia ampla
  • Pausa sinusal (RR longo)
  • Bloqueio AV de 2º grau (padrão Wenckebach / Mobitz II)

Estágio 2 – RandomForest sobre features morfológicas:
  • Treinado com anotações do MIT-BIH (se disponíveis)
  • Classifica cada batimento: Normal, PVC, APC, LBBB, RBBB, Paced…

Classes de arritmia detectadas (AAMI):
  N  – Normal
  V  – PVC (Premature Ventricular Contraction)
  A  – APC (Atrial Premature Contraction)
  L  – LBBB (Left Bundle Branch Block)
  R  – RBBB (Right Bundle Branch Block)
  /  – Paced
  F  – Fusion
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

# Importação condicional para não falhar se sklearn não instalado
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.exceptions import NotFittedError
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from .features import extract_beat_features, compute_hrv_metrics
from .loader import AAMI_MAP


# ──────────────────────────────────────────────────────────────
# Classificação por regras clínicas
# ──────────────────────────────────────────────────────────────

class RuleBasedClassifier:
    """
    Classificador determinístico baseado em limiares clínicos.

    Hierarquia de diagnósticos:
    ─────────────────────────────
    1. Pausa Sinusal          (RR > 2.0 s)
    2. PVC                    (RR prematuro + QRS largo estimado)
    3. APC                    (RR prematuro + QRS normal)
    4. Taquicardia (≥ 100 BPM)
    5. Bradicardia (≤ 50 BPM)
    6. FA / Flutter Atrial    (alta irregularidade de RR)
    7. Normal
    """

    # Limiares clínicos
    TACHY_BPM = 100.0
    BRADY_BPM = 50.0
    PAUSE_RR_MS = 2000.0          # pausa sinusal > 2 s
    PREMATURE_RATIO = 0.80        # RR < 80% da média local → prematuro
    WIDE_QRS_MS = 120.0           # QRS > 120 ms → ventricular ou bloqueio
    AF_RMSSD_THRESHOLD_MS = 50.0  # RMSSD alta → irregularidade (FA)
    AF_CV_THRESHOLD = 0.15        # coeficiente de variação dos RR > 15%

    def classify_beats(
        self,
        r_peaks: np.ndarray,
        fs: int,
        features: Optional[np.ndarray] = None,
    ) -> list[str]:
        """
        Classifica cada batimento com base em regras clínicas.

        Parâmetros
        ----------
        r_peaks  : posições dos picos R (amostras)
        fs       : frequência de amostragem
        features : matriz N×D de features (colunas 0-5 = RR, 13 = largura QRS)

        Retorna
        -------
        list[str] com rótulo para cada batimento
        """
        n = len(r_peaks)
        labels = ["N"] * n

        if n < 2:
            return labels

        rr_ms = np.diff(r_peaks) / fs * 1000.0
        # Estende para ter um RR por batimento
        rr_per_beat = np.concatenate([[rr_ms[0]], rr_ms])

        for i in range(n):
            rr = rr_per_beat[i]

            # RR médio local (±4 batimentos)
            lo = max(0, i - 4)
            hi = min(n - 1, i + 4)
            rr_local = rr_per_beat[lo:hi + 1]
            rr_mean = float(np.mean(rr_local))

            # Largura estimada do QRS (de features se disponível)
            qrs_width = 0.0
            if features is not None and i < len(features):
                qrs_width = float(features[i, 13])

            bpm = 60000.0 / rr if rr > 0 else 0.0

            # ── Regras em cascata ──────────────────────────────────
            if rr > self.PAUSE_RR_MS:
                labels[i] = "PAUSE"

            elif rr < self.PREMATURE_RATIO * rr_mean:
                # Batimento prematuro – diferencia PVC vs APC pela largura
                if qrs_width > self.WIDE_QRS_MS or qrs_width == 0.0:
                    labels[i] = "V"   # PVC (QRS largo ou desconhecido)
                else:
                    labels[i] = "A"   # APC (QRS estreito)

            elif bpm >= self.TACHY_BPM:
                labels[i] = "TACHY"

            elif bpm <= self.BRADY_BPM:
                labels[i] = "BRADY"

            # else → Normal (já inicializado)

        return labels

    def classify_rhythm(
        self, r_peaks: np.ndarray, fs: int
    ) -> dict[str, object]:
        """
        Classifica o ritmo global do registro (não beat-a-beat).

        Retorna dict com campos:
          - rhythm        : str (ex: "AF", "Normal Sinus", "Bradycardia")
          - mean_bpm      : float
          - is_regular    : bool
          - af_probability: float (0–1)
        """
        hrv = compute_hrv_metrics(r_peaks, fs)

        if len(r_peaks) < 3:
            return {"rhythm": "Insufficient data", "mean_bpm": 0.0,
                    "is_regular": True, "af_probability": 0.0}

        mean_bpm = hrv.get("mean_bpm", 0.0)
        rr_ms = hrv.get("rr_ms", np.array([]))
        rmssd = hrv.get("rmssd_ms", 0.0)

        cv = float(np.std(rr_ms) / np.mean(rr_ms)) if np.mean(rr_ms) > 0 else 0.0
        is_regular = cv < self.AF_CV_THRESHOLD

        # Estimativa simples de probabilidade de FA
        af_prob = min(1.0, cv / 0.5)

        if cv > self.AF_CV_THRESHOLD and rmssd > self.AF_RMSSD_THRESHOLD_MS:
            rhythm = "Atrial Fibrillation / Flutter"
        elif mean_bpm >= self.TACHY_BPM:
            rhythm = "Sinus Tachycardia"
        elif mean_bpm <= self.BRADY_BPM:
            rhythm = "Sinus Bradycardia"
        else:
            rhythm = "Normal Sinus Rhythm"

        return {
            "rhythm": rhythm,
            "mean_bpm": mean_bpm,
            "is_regular": is_regular,
            "af_probability": af_prob,
            "cv_rr": cv,
            "rmssd_ms": rmssd,
        }


# ──────────────────────────────────────────────────────────────
# Classificador ML (RandomForest)
# ──────────────────────────────────────────────────────────────

class MLBeatClassifier:
    """
    Classificador RandomForest para rotulação beat-a-beat.

    Treina com features extraídas por features.py e rótulos das
    anotações do especialista (MIT-BIH).

    Uso típico
    ----------
    clf = MLBeatClassifier()
    clf.fit(X_train, y_train)
    labels = clf.predict(X_test)
    """

    # Mapeamento de rótulos MIT-BIH para classes AAMI agrupadas
    LABEL_MAP = {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,   # Normal / LBBB / RBBB
        "A": 1, "a": 1, "S": 1, "J": 1,              # Supraventricular (APC)
        "V": 2, "E": 2,                               # Ventricular (PVC)
        "F": 3,                                       # Fusion
        "/": 4, "f": 4,                               # Paced
    }
    CLASS_NAMES = ["Normal", "APC", "PVC", "Fusion", "Paced"]

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn não encontrado. "
                "Execute: pip install scikit-learn"
            )
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )),
        ])
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "MLBeatClassifier":
        """Treina o classificador."""
        self.model.fit(X, y, rf__sample_weight=sample_weight)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classifica batimentos; retorna array de inteiros (índice de classe)."""
        if not self._fitted:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades por classe (N × n_classes)."""
        if not self._fitted:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")
        return self.model.predict_proba(X)

    def labels_from_annotations(
        self, ann_symbols: list[str]
    ) -> np.ndarray:
        """Converte símbolos MIT-BIH para inteiros de classe."""
        return np.array(
            [self.LABEL_MAP.get(s, -1) for s in ann_symbols], dtype=int
        )

    def save(self, path: str) -> None:
        """Serializa o modelo treinado com pickle."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[classifier] Modelo salvo → {path}")

    @classmethod
    def load(cls, path: str) -> "MLBeatClassifier":
        """Carrega modelo serializado."""
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[classifier] Modelo carregado ← {path}")
        return obj


# ──────────────────────────────────────────────────────────────
# Fachada unificada
# ──────────────────────────────────────────────────────────────

class BeatClassifier:
    """
    Fachada que combina regras clínicas + ML (quando treinado).

    Se o modelo ML não estiver treinado, usa apenas as regras.
    Se estiver treinado, combina: regras têm prioridade para PAUSE/TACHY/BRADY,
    ML decide entre N/V/A/F/Paced.
    """

    def __init__(self) -> None:
        self.rules = RuleBasedClassifier()
        self.ml: Optional[MLBeatClassifier] = (
            MLBeatClassifier() if _SKLEARN_AVAILABLE else None
        )
        self._ml_trained = False

    def train(
        self,
        signal: np.ndarray,
        r_peaks: np.ndarray,
        ann_symbols: list[str],
        fs: int,
    ) -> "BeatClassifier":
        """
        Treina o modelo ML com um registro anotado.

        Parâmetros
        ----------
        signal      : ECG filtrado
        r_peaks     : picos R detectados
        ann_symbols : símbolos das anotações (mesmo comprimento que r_peaks)
        fs          : frequência de amostragem
        """
        if self.ml is None:
            warnings.warn("sklearn não disponível; usando apenas regras.")
            return self

        X = extract_beat_features(signal, r_peaks, fs)
        y = self.ml.labels_from_annotations(ann_symbols)

        # Remove batimentos desconhecidos (y == -1)
        mask = y >= 0
        X, y = X[mask], y[mask]

        if len(np.unique(y)) < 2:
            warnings.warn("Menos de 2 classes nos dados de treino; ML ignorado.")
            return self

        self.ml.fit(X, y)
        self._ml_trained = True
        print(f"[classifier] RF treinado com {len(y)} batimentos "
              f"| {len(np.unique(y))} classes")
        return self

    def classify(
        self,
        signal: np.ndarray,
        r_peaks: np.ndarray,
        fs: int,
    ) -> tuple[list[str], dict]:
        """
        Classifica todos os batimentos e o ritmo global.

        Retorna
        -------
        (beat_labels, rhythm_info)
        beat_labels : list[str] com rótulo AAMI por batimento
        rhythm_info : dict com diagnóstico global do ritmo
        """
        features = extract_beat_features(signal, r_peaks, fs)

        # Sempre roda regras
        rule_labels = self.rules.classify_beats(r_peaks, fs, features)
        rhythm_info = self.rules.classify_rhythm(r_peaks, fs)

        final_labels = rule_labels.copy()

        # Sobrescreve com ML para batimentos "normais" (regras não detectaram nada especial)
        if self._ml_trained and self.ml is not None and len(r_peaks):
            ml_preds = self.ml.predict(features)
            class_names = MLBeatClassifier.CLASS_NAMES

            for i, rl in enumerate(rule_labels):
                if rl in ("N",):  # apenas onde regras dizem Normal
                    ml_cls = int(ml_preds[i])
                    if ml_cls < len(class_names):
                        # Mapeia de volta para símbolo AAMI
                        name = class_names[ml_cls]
                        if name == "APC":
                            final_labels[i] = "A"
                        elif name == "PVC":
                            final_labels[i] = "V"
                        elif name == "Fusion":
                            final_labels[i] = "F"
                        elif name == "Paced":
                            final_labels[i] = "/"
                        # Normal permanece "N"

        return final_labels, rhythm_info
