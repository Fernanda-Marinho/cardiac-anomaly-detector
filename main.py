import numpy as np
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks


def generate_synthetic_ecg(fs=250, duration=10):
    t = np.linspace(0, duration, fs * duration)
    heart_rate_pps = 1.2  
    ecg = np.sin(2 * np.pi * heart_rate_pps * t) ** 60  
    baseline_wander = 0.5 * np.sin(2 * np.pi * 0.1 * t)
    high_freq_noise = 0.05 * np.random.normal(size=len(t))
    return t, ecg + baseline_wander + high_freq_noise

def butter_bandpass_filter(data, lowcut=5.0, highcut=15.0, fs=250, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def pta_analysis_pipeline(signal, fs=250):
    filtered = butter_bandpass_filter(signal, lowcut=5.0, highcut=15.0, fs=fs)
    derivative = np.diff(filtered, prepend=filtered[0])
    squared = derivative ** 2 #quadrature
    window_len = int(0.12 * fs) 
    integrated = np.convolve(squared, np.ones(window_len)/window_len, mode='same')
    
    min_distance = int(0.4 * fs) 
    peaks, _ = find_peaks(integrated, distance=min_distance, prominence=np.mean(integrated)*0.5)
    
    return filtered, integrated, peaks


fs = 250  
t, raw_ecg = generate_synthetic_ecg(fs=fs)
filtered_ecg, processed_ecg, r_peaks = pta_analysis_pipeline(raw_ecg, fs=fs)

r_peak_times = t[r_peaks]
rr_intervals = np.diff(r_peak_times)
bpm = 60.0 / np.mean(rr_intervals)
print(f"Analysis Complete: Detected {len(r_peaks)} beats. Average Heart Rate: {bpm:.1f} BPM")

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, raw_ecg, label='Raw Corrupted ECG', color='gray')
plt.title('Stage 1: Raw Signal Input')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, filtered_ecg, label='Bandpass Filtered (5-15 Hz)', color='blue')
plt.title('Stage 2: Noise Removal')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, processed_ecg, label='PTA Integrated Signal', color='orange')
plt.plot(t[r_peaks], processed_ecg[r_peaks], "x", color='red', label='Tracked R-Peaks', markersize=10)
plt.title(f'Stage 3: Peak Tracking & Integration (Estimated HR: {bpm:.1f} BPM)')
plt.xlabel('Time (seconds)')
plt.legend()
plt.tight_layout()
plt.savefig("output/ecg_analysis.png", dpi=150, bbox_inches='tight')
print("Plot saved to ecg_analysis.png")
