import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ================= Задание 1 =================
# Загрузка изображения фразы
input_image_path = "lab6/phrase.bmp"
output_dir = "lab6_output"
os.makedirs(output_dir, exist_ok=True)

img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError(f"Не найден файл {input_image_path}")

# Бинаризация
_, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

# ================= Задание 2 =================
# Построение горизонтального и вертикального профилей

# Горизонтальный профиль
horizontal_profile = np.sum(img_bin, axis=1)

# Вертикальный профиль
vertical_profile = np.sum(img_bin, axis=0)

# Сохраним графики профилей
plt.figure(figsize=(10,4))
plt.plot(horizontal_profile)
plt.title('Горизонтальный профиль изображения')
plt.xlabel('Строка')
plt.ylabel('Сумма чёрных пикселей')
plt.grid()
plt.savefig(os.path.join(output_dir, 'horizontal_profile.png'))
plt.close()

plt.figure(figsize=(10,4))
plt.plot(vertical_profile)
plt.title('Вертикальный профиль изображения')
plt.xlabel('Столбец')
plt.ylabel('Сумма чёрных пикселей')
plt.grid()
plt.savefig(os.path.join(output_dir, 'vertical_profile.png'))
plt.close()

# ================= Задание 3 =================
# Сегментация символов в строках через профили с прореживанием

def find_intervals(profile, threshold=5):
    intervals = []
    start = None
    for i, value in enumerate(profile):
        if value > threshold and start is None:
            start = i
        elif value <= threshold and start is not None:
            intervals.append((start, i))
            start = None
    if start is not None:
        intervals.append((start, len(profile)))
    return intervals

lines = find_intervals(horizontal_profile)

# Для обводки символов
img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for y1, y2 in lines:
    line_img = img_bin[y1:y2, :]
    vertical_profile_line = np.sum(line_img, axis=0)
    chars = find_intervals(vertical_profile_line)

    for x1, x2 in chars:
        # Рисуем прямоугольник на исходном цветном изображении
        cv2.rectangle(img_colored, (x1, y1), (x2, y2), (0, 0, 255), 1)

# Сохраняем изображение с обведёнными символами
cv2.imwrite(os.path.join(output_dir, 'segmented_phrase.png'), img_colored)

# ================= Задание 4 =================
# Построение профилей символов

# Берём первый найденный символ как пример
first_line_y1, first_line_y2 = lines[0]
first_line_img = img_bin[first_line_y1:first_line_y2, :]
first_vertical_profile = np.sum(first_line_img, axis=0)
first_char_x1, first_char_x2 = find_intervals(first_vertical_profile)[0]
first_char_img = first_line_img[:, first_char_x1:first_char_x2]

# Профили первого символа
char_horizontal_profile = np.sum(first_char_img, axis=1)
char_vertical_profile = np.sum(first_char_img, axis=0)

plt.figure(figsize=(10,4))
plt.plot(char_horizontal_profile)
plt.title('Горизонтальный профиль примера символа')
plt.xlabel('Строка')
plt.ylabel('Сумма чёрных пикселей')
plt.grid()
plt.savefig(os.path.join(output_dir, 'char_horizontal_profile.png'))
plt.close()

plt.figure(figsize=(10,4))
plt.plot(char_vertical_profile)
plt.title('Вертикальный профиль примера символа')
plt.xlabel('Столбец')
plt.ylabel('Сумма чёрных пикселей')
plt.grid()
plt.savefig(os.path.join(output_dir, 'char_vertical_profile.png'))
plt.close()

# ================= Задание 5 (дополнительно) =================
# Выделение прямоугольника для всего текста (обрамляющий прямоугольник)

ys, xs = np.nonzero(img_bin)
y_min, y_max = np.min(ys), np.max(ys)
x_min, x_max = np.min(xs), np.max(xs)
img_total_bounding = img_colored.copy()
cv2.rectangle(img_total_bounding, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Сохраняем общий прямоугольник
cv2.imwrite(os.path.join(output_dir, 'bounding_whole_text.png'), img_total_bounding)

# ================= Конец =================
print("✅ Лабораторная работа №6 выполнена полностью. Результаты в папке lab6_output.")
