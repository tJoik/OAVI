from PIL import Image, ImageDraw, ImageFont 
import os
import numpy as np
import csv
import matplotlib.pyplot as plt

output_folder = "symbols_test"
os.makedirs(output_folder, exist_ok=True)

font_path = "timesi.ttf"
if not os.path.exists(font_path):
    print("Шрифт Times New Roman не найден.")
    exit()

font_size = 52
font = ImageFont.truetype(font_path, font_size)
characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

features_file = os.path.join(output_folder, "features.csv")
with open(features_file, mode="w", newline="") as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow([
        "char",
        "q1_abs", "q2_abs", "q3_abs", "q4_abs",
        "q1_rel", "q2_rel", "q3_rel", "q4_rel",
        "center_x", "center_y", "center_x_rel", "center_y_rel",
        "ix", "iy", "ix_rel", "iy_rel"
    ])

    for char in characters:
        image_size = (1000, 1000)  
        image = Image.new("L", image_size, 255)  
        draw = ImageDraw.Draw(image)

        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        image = Image.new("L", (text_width + 40, text_height + 40), 255)
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), char, font=font, fill=0)

        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)

        img_array = np.array(image)
        binary_img = (img_array < 128).astype(int)

        h, w = binary_img.shape
        h2, w2 = h // 2, w // 2

        q1 = binary_img[:h2, :w2]
        q2 = binary_img[:h2, w2:]
        q3 = binary_img[h2:, :w2]
        q4 = binary_img[h2:, w2:]

        q1_abs = np.sum(q1)
        q2_abs = np.sum(q2)
        q3_abs = np.sum(q3)
        q4_abs = np.sum(q4)

        q_area = q1.size

        q1_rel = q1_abs / q_area
        q2_rel = q2_abs / q_area
        q3_rel = q3_abs / q_area
        q4_rel = q4_abs / q_area

        weight = np.sum(binary_img)
        if weight == 0:
            cx = cy = ix = iy = ix_rel = iy_rel = 0
            cx_rel = cy_rel = 0
            horizontal_profile = vertical_profile = [0]
        else:
            y_indices, x_indices = np.indices(binary_img.shape)

            # Центр тяжести
            cx = np.sum(x_indices * binary_img) / weight
            cy = np.sum(y_indices * binary_img) / weight
            cx_rel = (cx - 1) / (w - 1) if w > 1 else 0
            cy_rel = (cy - 1) / (h - 1) if h > 1 else 0

            # Осевые моменты
            ix = np.sum(((y_indices - cy) ** 2) * binary_img)
            iy = np.sum(((x_indices - cx) ** 2) * binary_img)

            # Нормированные моменты
            ix_rel = ix / (w ** 2 * h ** 2)
            iy_rel = iy / (w ** 2 * h ** 2)

            # Профили
            horizontal_profile = np.sum(binary_img, axis=1)
            vertical_profile = np.sum(binary_img, axis=0)

        # Запись скалярных признаков в CSV
        writer.writerow([
            char,
            q1_abs, q2_abs, q3_abs, q4_abs,
            round(q1_rel, 4), round(q2_rel, 4), round(q3_rel, 4), round(q4_rel, 4),
            round(cx, 2), round(cy, 2),
            round(cx_rel, 4), round(cy_rel, 4),
            round(ix, 2), round(iy, 2),
            round(ix_rel, 6), round(iy_rel, 6)
        ])

        # Сохраняем изображение символа
        image.save(os.path.join(output_folder, f"char_{ord(char)}.png"))

        # Сохраняем PNG-графики профилей
        # Горизонтальный профиль (ось Y)
        plt.figure(figsize=(4, 2))
        plt.bar(range(len(horizontal_profile)), horizontal_profile)
        plt.title(f"Horizontal profile - {char}")
        plt.xlabel("Y")
        plt.ylabel("Sum")
        plt.xticks(ticks=np.arange(len(horizontal_profile)), labels=np.arange(len(horizontal_profile)), fontsize=6)
        plt.yticks(fontsize=6)
        plt.grid(axis='y', linestyle=':', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"profile_horizontal_{char}.png"))
        plt.close()

        # Вертикальный профиль (ось X)
        plt.figure(figsize=(4, 2))
        plt.bar(range(len(vertical_profile)), vertical_profile)
        plt.title(f"Vertical profile - {char}")
        plt.xlabel("X")
        plt.ylabel("Sum")
        plt.xticks(ticks=np.arange(len(vertical_profile)), labels=np.arange(len(vertical_profile)), fontsize=6)
        plt.yticks(fontsize=6)
        plt.grid(axis='y', linestyle=':', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"profile_vertical_{char}.png"))
        plt.close()

print("✅ Все признаки и профили успешно сохранены.")
