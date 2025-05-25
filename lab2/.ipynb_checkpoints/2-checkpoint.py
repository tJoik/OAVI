import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === Устанавливаем рабочую директорию на папку, где находится этот скрипт ===
script_dir = os.path.dirname(os.path.abspath(__file__))  # Папка, где лежит 2.py
os.chdir(script_dir)  # Меняем рабочую директорию
print(f"Рабочая директория изменена на: {os.getcwd()}")  # Для отладки

# === Пути к файлам ===
image_filename = "1map.png"  # Оригинальный файл
image_path = os.path.join(script_dir, image_filename)  # Полный путь к файлу
output_halftone = os.path.join(script_dir, "1map_halftone.png")  # Полутоновое изображение
output_thresholding = os.path.join(script_dir, "1map_thresholding.png")  # Бинаризированное изображение

# === Проверяем, существует ли изображение ===
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Ошибка: изображение '{image_filename}' не найдено в {script_dir}!")

# === Загружаем изображение ===
image = cv2.imread(image_path)

# === Проверяем, успешно ли загружено изображение ===
if image is None:
    raise ValueError(f"Ошибка: файл '{image_filename}' повреждён или имеет неподдерживаемый формат!")

# === Преобразуем изображение в оттенки серого (полутоновое) ===
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# === Функция адаптивной бинаризации Феня и Тана ===
def fen_tan_binarization(image, window_size=5, C=5):
    mean = cv2.boxFilter(image, ddepth=-1, ksize=(window_size, window_size))
    sq_mean = cv2.boxFilter(image**2, ddepth=-1, ksize=(window_size, window_size))
    stddev = np.sqrt(sq_mean - mean**2)  # Вычисление стандартного отклонения

    threshold = mean + C * stddev  # Порог с учетом стандартного отклонения
    binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)  # Бинаризация

    return binary_image

# === Применяем адаптивную бинаризацию Феня и Тана (окно 5×5) ===
binary_fen_tan = fen_tan_binarization(gray_image, window_size=5, C=5)

# === Сохраняем обработанные изображения ===
cv2.imwrite(output_halftone, gray_image)  # Полутоновое
cv2.imwrite(output_thresholding, binary_fen_tan)  # Бинаризированное

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
