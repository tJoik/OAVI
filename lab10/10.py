import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
import os

# === Зафиксировать рабочую директорию как папку, где лежит скрипт ===
os.chdir(os.path.dirname(__file__))

# === Пути к файлам ===
files = {
    "AAA": "AAA.wav",
    "III": "III.wav",
    "GAV": "GAV.wav"
}

output_dir = "spectrograms"
os.makedirs(output_dir, exist_ok=True)

# === Функция обрезки начала сигнала ===
def cut_leading_seconds(data, rate, seconds=1.0):
    start_sample = int(seconds * rate)
    return data[start_sample:]

# === Задание 2: спектрограмма ===
def process_and_plot_spectrogram(filepath, label):
    rate, data = wav.read(filepath)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / np.max(np.abs(data))
    data = cut_leading_seconds(data, rate)

    window = signal.windows.hann(1024)
    f, t, Sxx = signal.spectrogram(data, fs=rate, window=window, nperseg=1024, noverlap=512, scaling='density', mode='magnitude')
    Sxx_log = 10 * np.log10(Sxx + 1e-10)

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, Sxx_log, shading='gouraud')
    plt.title(f"Спектрограмма сигнала: {label}")
    plt.xlabel("Время [с]")
    plt.ylabel("Частота [Гц]")
    plt.colorbar(label="Интенсивность [дБ]")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"spectrogram_{label}.png"))
    plt.close()

# === Задание 3: мин/макс частоты ===
def plot_min_max_freq(filename, label):
    rate, data = wav.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / np.max(np.abs(data))
    data = cut_leading_seconds(data, rate)

    spectrum = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), d=1/rate)
    power = np.abs(spectrum)

    threshold = np.max(power) * 0.01
    significant_freqs = freqs[power > threshold]
    min_f = np.min(significant_freqs)
    max_f = np.max(significant_freqs)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, power)
    plt.axvline(min_f, color='green', linestyle='--', label=f"Мин: {min_f:.1f} Гц")
    plt.axvline(max_f, color='red', linestyle='--', label=f"Макс: {max_f:.1f} Гц")
    plt.title(f"Задание 3: Частоты сигнала {label}")
    plt.xlabel("Частота [Гц]")
    plt.ylabel("Амплитуда")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"3_{label}_min_max_freq.png"))
    plt.close()

# === Задание 4: основной тон и обертоны ===
def plot_harmonics(filename, label):
    rate, data = wav.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / np.max(np.abs(data))
    data = cut_leading_seconds(data, rate)

    window_size = int(rate * 0.1)
    if len(data) < window_size:
        segment = data
    else:
        start = len(data) // 2 - window_size // 2
        segment = data[start:start + window_size]

    window = signal.windows.hann(len(segment))
    segment = segment * window
    spectrum = np.fft.rfft(segment)
    freqs = np.fft.rfftfreq(len(segment), d=1/rate)
    power = np.abs(spectrum) ** 2

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, power)
    plt.title(f"Задание 4: Обертоны сигнала {label}")
    plt.xlabel("Частота [Гц]")
    plt.ylabel("Энергия")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"4_{label}_harmonics.png"))
    plt.close()

# === Задание 5: три форманты + среднее ===
def plot_formants(filename, label):
    rate, data = wav.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / np.max(np.abs(data))
    data = cut_leading_seconds(data, rate)

    window_size = int(rate * 0.1)
    step = window_size
    freqs = np.fft.rfftfreq(window_size, d=1 / rate)

    all_formants = []
    time_stamps = []

    for start in range(0, len(data) - window_size, step):
        segment = data[start:start + window_size]
        spectrum = np.fft.rfft(segment * signal.windows.hann(window_size))
        power = np.abs(spectrum) ** 2
        max_idx = np.argsort(power)[-3:]
        all_formants.append(freqs[max_idx])
        time_stamps.append((start + rate) / rate)  # сдвиг на 1с

    all_formants = np.array(all_formants)
    avg_all = np.mean(all_formants)

    plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.plot(time_stamps, all_formants[:, i], label=f"Форманта {i+1}")
    plt.axhline(avg_all, color='black', linestyle='--', linewidth=2, label=f"Общее среднее: {avg_all:.1f} Гц")
    plt.title(f"Задание 5: Форманты сигнала {label}")
    plt.xlabel("Время [с]")
    plt.ylabel("Частота [Гц]")
    plt.legend()
    plt.grid()

    plt.text(0.95, 0.95, f"Общее среднее: {avg_all:.1f} Гц", transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"5_{label}_formants.png"))
    plt.close()

# === Задание 5.1: одна линия — сумма всех трёх формант ===
def plot_formants_sum(filename, label):
    rate, data = wav.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / np.max(np.abs(data))
    data = cut_leading_seconds(data, rate)

    window_size = int(rate * 0.1)
    step = window_size
    freqs = np.fft.rfftfreq(window_size, d=1 / rate)

    summed_formants = []
    time_stamps = []

    for start in range(0, len(data) - window_size, step):
        segment = data[start:start + window_size]
        spectrum = np.fft.rfft(segment * signal.windows.hann(window_size))
        power = np.abs(spectrum) ** 2
        max_idx = np.argsort(power)[-3:]
        formant_sum = np.sum(freqs[max_idx])
        summed_formants.append(formant_sum)
        time_stamps.append((start + rate) / rate)  # сдвиг на 1с

    avg_sum = np.mean(summed_formants)

    plt.figure(figsize=(10, 5))
    plt.plot(time_stamps, summed_formants, color='blue', label='Сумма 3 формант')
    plt.title(f"Задание 5.1: Суммарная форманта сигнала {label}")
    plt.xlabel("Время [с]")
    plt.ylabel("Суммарная частота [Гц]")
    plt.grid()
    plt.legend()

    plt.text(0.95, 0.95, f"Среднее: {avg_sum:.1f} Гц", transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"5_1_{label}_formants_sum.png"))
    plt.close()

# === Запуск всех заданий ===
for label, filepath in files.items():
    process_and_plot_spectrogram(filepath, label)
    plot_min_max_freq(filepath, label)
    plot_harmonics(filepath, label)
    plot_formants(filepath, label)       # Задание 5
    plot_formants_sum(filepath, label)   # Задание 5.1

print("✅ Все задания выполнены. Графики в папке:", output_dir)
