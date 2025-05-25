import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Параметры LBP
radius = 1
n_points = 8 * radius
method = 'uniform'

def compute_lbp_histogram(image, n_bins):
    lbp = local_binary_pattern(image, n_points, radius, method)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    return lbp, hist

def process_image(image_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"result_{timestamp}"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print(f"[!] Не удалось открыть файл: {image_path}")
        return

    # Перевод в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Контрастирование (выравнивание гистограммы)
    equalized = cv2.equalizeHist(gray)

    # Расчёт LBP и гистограмм
    n_bins = int(n_points + 2)  # для 'uniform' паттерна

    lbp_gray, hist_gray = compute_lbp_histogram(gray, n_bins)
    lbp_eq, hist_eq = compute_lbp_histogram(equalized, n_bins)

    # Нормировка гистограмм
    hist_gray = hist_gray.astype(np.float64) / hist_gray.sum()
    hist_eq = hist_eq.astype(np.float64) / hist_eq.sum()

    # Визуализация
    fig, axs = plt.subplots(4, 2, figsize=(14, 20))
    fig.suptitle("ЛР8. Вариант 23. Текстурный анализ (LBP)", fontsize=16)

    # 1. Исходное полутоновое изображение
    axs[0, 0].imshow(gray, cmap='gray')
    axs[0, 0].set_title("Исходное изображение (серое)")
    axs[0, 0].axis('off')

    # 2. Контрастированное полутоновое изображение
    axs[0, 1].imshow(equalized, cmap='gray')
    axs[0, 1].set_title("После выравнивания гистограммы")
    axs[0, 1].axis('off')

    # 3. Гистограмма исходного изображения
    axs[1, 0].hist(gray.ravel(), bins=256, color='gray')
    axs[1, 0].set_title("Гистограмма яркости (до контрастирования)")
    axs[1, 0].set_xlabel("Яркость")
    axs[1, 0].set_ylabel("Частота")

    # 4. Гистограмма контрастированного изображения
    axs[1, 1].hist(equalized.ravel(), bins=256, color='gray')
    axs[1, 1].set_title("Гистограмма яркости (после контрастирования)")
    axs[1, 1].set_xlabel("Яркость")
    axs[1, 1].set_ylabel("Частота")

    # 5. Матрица LBP для исходного изображения
    axs[2, 0].imshow(lbp_gray, cmap='gray')
    axs[2, 0].set_title("Матрица LBP (до контрастирования)")
    axs[2, 0].axis('off')

    # 6. Матрица LBP для контрастированного изображения
    axs[2, 1].imshow(lbp_eq, cmap='gray')
    axs[2, 1].set_title("Матрица LBP (после контрастирования)")
    axs[2, 1].axis('off')

    # 7. Гистограмма H(LBP)
    axs[3, 0].bar(np.arange(n_bins) - 0.2, hist_gray, width=0.4, label="До", alpha=0.7)
    axs[3, 0].bar(np.arange(n_bins) + 0.2, hist_eq, width=0.4, label="После", alpha=0.7)
    axs[3, 0].set_title("Гистограмма H(LBP)")
    axs[3, 0].set_xlabel("Бины LBP")
    axs[3, 0].set_ylabel("Частота")
    axs[3, 0].legend()

    # 8. Пустая ячейка
    axs[3, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_full_report.png"))
    plt.close()

    print(f"[✓] Готово! Результаты сохранены в папку '{output_dir}'.")

# === ВАЖНО ===
# Передаём правильный путь к изображению (с учётом структуры папок)
process_image("lab8/images/texture.png")
