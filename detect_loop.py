import sys
from pathlib import Path

# Adiciona a pasta atual ao path para encontrar o módulo src
sys.path.insert(0, str(Path(__file__).parent))

from src.loader import load_record, list_records
from src.pan_tompkins import pan_tompkins
from src.classifier import BeatClassifier

def main():
    # Obtém a lista completa (100 a 234) definida no loader.py
    records = list_records()
    classifier = BeatClassifier()
    
    print(f"{'ID':<6} | {'Status/Arritmias Detectadas'}")
    print("-" * 50)

    for record in records:
        try:
            # Carrega 30s do sinal (frequência padrão 360Hz)
            signal, fs, _, _ = load_record(record, duration_sec=30)
            
            # Etapa 1: Detecção de Picos R (Pan-Tompkins)
            results = pan_tompkins(signal, fs)
            r_peaks = results["r_peaks"]
            
            # Etapa 2: Classificação (O erro estava aqui: remova o 'X')
            # A classe BeatClassifier já importa extract_beat_features internamente
            labels, rhythm_info = classifier.classify(signal, r_peaks, fs)
            
            # Filtra o que não é "Normal" (N)
            anomalies = [label for label in labels if label != "Normal"]
            
            # Identifica problemas de ritmo (Tachy, Brady, FA, Pausa)
            rhythm_anomalies = [k for k, v in rhythm_info.items() if v is True]

            if anomalies or rhythm_anomalies:
                # Agrupa a contagem de batimentos anômalos
                summary = {a: anomalies.count(a) for a in set(anomalies)}
                output = f"{record:<6} | "
                
                if rhythm_anomalies:
                    output += f"RITMO: {', '.join(rhythm_anomalies)} | "
                
                output += ", ".join([f"{k}: {v}" for k, v in summary.items()])
                print(output)
            else:
                # Opcional: descomente para ver registros limpos
                # print(f"{record:<6} | Normal")
                pass

        except Exception as e:
            print(f"{record:<6} | Erro: {e}")

if __name__ == "__main__":
    main()