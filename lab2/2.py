import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === Устанавливаем рабочую директорию ===
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Рабочая директория: {os.getcwd()}")

# === Пути к файлам ===
image_filename = "6text.png"
image_path = os.path.join(script_dir, image_filename)
output_halftone = os.path.join(script_dir, "6text_halftone.png")
output_thresholding = os.path.join(script_dir, "6text_thresholding.png")

# === Проверяем, существует ли изображение ===
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Ошибка: изображение '{image_filename}' не найдено!")

# === Загружаем изображение ===
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Ошибка: файл '{image_filename}' повреждён или имеет неподдерживаемый формат!")

# === Преобразуем в полутоновое ===
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# === Функция адаптивной бинаризации Феня и Тана ===
def fen_tan_binarization(image, window_size=5, alpha1=0.15, gamma=2, k1=0.2, k2=0.02):
    mean = cv2.boxFilter(image, ddepth=-1, ksize=(window_size, window_size))  # Среднее значение
    sq_mean = cv2.boxFilter(image**2, ddepth=-1, ksize=(window_size, window_size))
    stddev = np.sqrt(sq_mean - mean**2)  # Стандартное отклонение

    # Корректный расчёт максимального значения в окне
    M = cv2.dilate(image, np.ones((window_size, window_size), np.uint8))

    # Динамический диапазон (максимальная интенсивность пикселя)
    R = np.max(image)

    # Нормализация стандартного отклонения
    S_norm = stddev / R

    # Вычисляем alpha2 и alpha3 для каждого пикселя
    alpha2 = k1 * (S_norm ** gamma)
    alpha3 = k2 * (S_norm ** gamma)

    # Полная формула Феня и Тана
    threshold = (1 - alpha1) * mean + alpha2 * (stddev / R) * (mean - M) + alpha3 * M
    binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)

    return binary_image

# === Применяем адаптивную бинаризацию Феня и Тана (окно 5×5) ===
binary_fen_tan = fen_tan_binarization(gray_image, window_size=5, alpha1=0.15, gamma=2, k1=0.2, k2=0.02)

# === Сохраняем обработанные изображения ===
cv2.imwrite(output_halftone, gray_image)
cv2.imwrite(output_thresholding, binary_fen_tan)

print(f"✅ Полутоновое изображение сохранено как: {output_halftone}")
print(f"✅ Бинаризированное изображение сохранено как: {output_thresholding}")

# === Отображаем результат ===
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray_image, cmap="gray")
axes[0].set_title("Полутоновое изображение")
axes[0].axis("off")

axes[1].imshow(binary_fen_tan, cmap="gray")
axes[1].set_title("Бинаризация Феня и Тана (5×5)")
axes[1].axis("off")

plt.show()
