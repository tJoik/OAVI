import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# === Параметры ===
filename = r"C:\Users\shigr\Desktop\ОАИ\lab9\guitar.wav"
output_image_energy = "energy_40_50Hz.png"

# === Загрузка аудиофайла ===
rate, data = wav.read(filename)

# Переводим в моно, если стерео
if len(data.shape) > 1:
    data = data[:, 0]

# Нормализация
data = data.astype(np.float32)
data = data / np.max(np.abs(data))

# === Параметры анализа энергии ===
window_duration_sec = 0.1
window_size = int(rate * window_duration_sec)
step_size = window_size  # без перекрытия
frequencies = np.fft.rfftfreq(window_size, d=1 / rate)

energy_by_window = []
time_stamps = []

for start in range(0, len(data) - window_size, step_size):
    end = start + window_size
    segment = data[start:end]

    # БПФ и спектральная мощность
    spectrum = np.fft.rfft(segment)
    power_spectrum = np.abs(spectrum) ** 2

    # Энергия в диапазоне 40–50 Гц
    mask = (frequencies >= 40) & (frequencies <= 50)
    energy = np.sum(power_spectrum[mask])
    energy_by_window.append(energy)
    time_stamps.append(start / rate)

# === Визуализация ===
plt.figure(figsize=(10, 5))
plt.plot(time_stamps, energy_by_window, label='Энергия 40–50 Гц', color='blue')
plt.xlabel('Время [с]')
plt.ylabel('Энергия')
plt.title('Анализ энергии в диапазоне 40–50 Гц')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_image_energy)
plt.show()

print(f"✅ График энергии сохранён в файл: {output_image_energy}")
