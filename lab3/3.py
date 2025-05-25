import cv2
import numpy as np
import os

# Введите название изображения (замени на свое)
image_name = "hieroglyph.png"  
image_path = os.path.join(os.getcwd(), image_name)

# Проверяем, существует ли файл
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Файл '{image_name}' не найден в текущей папке!")

# Загружаем изображение (цветное или градации серого)
image_color = cv2.imread(image_path)  # Оригинал (на случай цветного)
image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Черно-белый вариант

# Определяем, является ли изображение бинарным (черно-белым)
_, binary_check = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
is_binary = np.array_equal(image_gray, binary_check)

# Если изображение полутоновое, нормализуем его
if not is_binary:
    image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Структурирующий элемент "кольцо" (3x3)
kernel = np.array([[1, 1, 1], 
                   [1, 0, 1], 
                   [1, 1, 1]], dtype=np.uint8)

# Выполняем морфологическое закрытие
closed_image = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel)

# Генерируем разностное изображение
if is_binary:
    diff_image = cv2.bitwise_xor(image_gray, closed_image)  # XOR для бинарных
else:
    diff_image = cv2.absdiff(image_gray, closed_image)  # Модуль разности для полутоновых

# Если исходное изображение было цветным, применяем к нему результат
if len(image_color.shape) == 3:
    color_closed = cv2.cvtColor(closed_image, cv2.COLOR_GRAY2BGR)
    color_diff = cv2.cvtColor(diff_image, cv2.COLOR_GRAY2BGR)
    filtered_color = cv2.bitwise_and(image_color, color_closed)  # Цветное с учетом закрытия
    diff_color = cv2.bitwise_and(image_color, color_diff)  # Цветное с учетом разницы
else:
    filtered_color, diff_color = closed_image, diff_image  # Оставляем черно-белое

# Генерируем пути сохранения
closed_name = "closed_" + image_name
diff_name = "difference_" + image_name
color_closed_name = "filtered_color_" + image_name
color_diff_name = "diff_color_" + image_name

closed_path = os.path.join(os.getcwd(), closed_name)
diff_path = os.path.join(os.getcwd(), diff_name)
color_closed_path = os.path.join(os.getcwd(), color_closed_name)
color_diff_path = os.path.join(os.getcwd(), color_diff_name)

# Сохраняем изображения
cv2.imwrite(closed_path, closed_image)
cv2.imwrite(diff_path, diff_image)
cv2.imwrite(color_closed_path, filtered_color)
cv2.imwrite(color_diff_path, diff_color)

print(f"✅ Фильтрованное изображение сохранено: {closed_path}")
print(f"✅ Разностное изображение сохранено: {diff_path}")
print(f"✅ Цветное фильтрованное изображение сохранено: {color_closed_path}")
print(f"✅ Цветное разностное изображение сохранено: {color_diff_path}")


