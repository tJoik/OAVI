import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cdist

# Пути
features_path = "C:/Users/shigr/Desktop/ОАИ/lab7/features.csv"
image_path = "C:/Users/shigr/Desktop/ОАИ/lab7/phrase.bmp"
output_dir = "C:/Users/shigr/Desktop/ОАИ/lab7/output_recognition"
os.makedirs(output_dir, exist_ok=True)

# === ЗАДАНИЕ 1: расчет евклидовых расстояний между эталонами ===
df = pd.read_csv(features_path, sep=";")
df["mass"] = df["q1_rel"] + df["q2_rel"] + df["q3_rel"] + df["q4_rel"]
etalon_features = df[["mass", "center_x_rel", "center_y_rel", "ix_rel", "iy_rel"]].to_numpy()
etalon_labels = df["char"].astype(str).to_numpy()

distance_matrix = cdist(etalon_features, etalon_features, metric="euclidean")
distances_df = pd.DataFrame(distance_matrix, index=etalon_labels, columns=etalon_labels)
distances_df.to_csv(os.path.join(output_dir, "distances.csv"), sep=";")

plt.figure(figsize=(12, 10))
plt.imshow(distance_matrix, cmap="viridis")
plt.colorbar(label="Евклидово расстояние")
plt.xticks(ticks=np.arange(len(etalon_labels)), labels=etalon_labels)
plt.yticks(ticks=np.arange(len(etalon_labels)), labels=etalon_labels)
plt.title("Матрица евклидовых расстояний между эталонами")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "distance_matrix.png"))
plt.close()

# === ЗАДАНИЯ 2–4: честное распознавание символов на основе ближайшего эталона ===
# Предположим, что phrase_features.csv — это заранее извлеченные признаки каждого символа из изображения
phrase_features_path = "C:/Users/shigr/Desktop/ОАИ/lab7/phrase_features.csv"
phrase_df = pd.read_csv(phrase_features_path, sep=";")
phrase_df["mass"] = phrase_df["q1_rel"] + phrase_df["q2_rel"] + phrase_df["q3_rel"] + phrase_df["q4_rel"]
phrase_features = phrase_df[["mass", "center_x_rel", "center_y_rel", "ix_rel", "iy_rel"]].to_numpy()

recognized = []

for feature_vector in phrase_features:
    dists = cdist([feature_vector], etalon_features, metric="euclidean")[0]
    top_indices = np.argsort(dists)[:9]  # Топ 9 гипотез
    hypotheses = [(etalon_labels[i], round(1 / (1 + dists[i]), 4)) for i in top_indices]
    hypotheses = sorted(hypotheses, key=lambda x: -x[1])
    recognized.append(hypotheses)

# Сохранение гипотез
output_txt = os.path.join(output_dir, "recognized_hypotheses.txt")
with open(output_txt, "w", encoding="utf-8") as f:
    for i, hyp in enumerate(recognized, 1):
        f.write(f"{i}: {[(char, score) for char, score in hyp]}\n")

# === ЗАДАНИЕ 4: вывод строки ===
top_hypotheses = [h[0][0] for h in recognized]
recognized_string = "".join(top_hypotheses)

comparison_path = os.path.join(output_dir, "final_result.txt")
with open(comparison_path, "w", encoding="utf-8") as f:
    f.write("Распознанная строка:\n")
    f.write(recognized_string + "\n\n")

# === ЗАДАНИЕ 5: если есть эталонная строка, сравниваем ===
expected_string = "LOREMIPSUMDOLORSITAMETCONSECTETURADIPISICINGELIT"
if len(expected_string) == len(recognized_string):
    num_total = len(expected_string)
    num_correct = sum(a == b for a, b in zip(recognized_string, expected_string))
    num_errors = num_total - num_correct
    accuracy = (num_correct / num_total) * 100

    with open(comparison_path, "a", encoding="utf-8") as f:
        f.write("Ожидаемая строка:\n")
        f.write(expected_string + "\n\n")
        f.write(f"Количество ошибок: {num_errors}\n")
        f.write(f"Доля верно распознанных символов: {accuracy:.2f}%\n")

    print(f"✅ Готово! Точность распознавания: {accuracy:.2f}%")
else:
    print("⚠️ Предупреждение: длина распознанной строки не совпадает с эталонной.")

# === Сохранение изображения фразы ===
image = Image.open(image_path).convert("L")
binary = np.array(image) < 128

plt.figure(figsize=(12, 5))
plt.imshow(binary, cmap="gray")
plt.title("Фраза для распознавания")
plt.axis("off")
plt.savefig(os.path.join(output_dir, "phrase_preview.png"))
plt.close()
