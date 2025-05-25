import numpy as np
import cv2
import os


# Меняем рабочую директорию на папку, где лежит этот скрипт
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# Функция загрузки изображения
def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

# Функция сохранения изображения
def save_image(image, filename):
    cv2.imwrite(filename, image)

# 1.1 Разбиение изображения на каналы R, G, B
def split_rgb(image):
    B, G, R = cv2.split(image)
    save_image(R, "1.1_R_channel.png")
    save_image(G, "1.1_G_channel.png")
    save_image(B, "1.1_B_channel.png")
    
    R_colored = np.zeros_like(image)
    R_colored[:, :, 2] = R
    G_colored = np.zeros_like(image)
    G_colored[:, :, 1] = G
    B_colored = np.zeros_like(image)
    B_colored[:, :, 0] = B
    
    save_image(R_colored, "1.1_R_colored.png")
    save_image(G_colored, "1.1_G_colored.png")
    save_image(B_colored, "1.1_B_colored.png")
    
    return R, G, B

# 1.2 Преобразование RGB в HSL и сохранение яркостной компоненты
def rgb_to_hsl(image):
    B, G, R = image[:,:,0] / 255.0, image[:,:,1] / 255.0, image[:,:,2] / 255.0
    C_max = np.maximum(R, np.maximum(G, B))
    C_min = np.minimum(R, np.minimum(G, B))
    delta = C_max - C_min
    
    # Вычисление оттенка (H)
    H = np.zeros_like(C_max)
    mask = delta != 0
    
    H[mask & (C_max == R)] = 60 * (((G[mask & (C_max == R)] - B[mask & (C_max == R)]) / delta[mask & (C_max == R)]) % 6)
    H[mask & (C_max == G)] = 60 * (((B[mask & (C_max == G)] - R[mask & (C_max == G)]) / delta[mask & (C_max == G)]) + 2)
    H[mask & (C_max == B)] = 60 * (((R[mask & (C_max == B)] - G[mask & (C_max == B)]) / delta[mask & (C_max == B)]) + 4)
    
    # Вычисление насыщенности (S)
    S = np.zeros_like(C_max)
    S[C_max != 0] = delta[C_max != 0] / C_max[C_max != 0]
    
    # Вычисление яркости (L)
    L = (C_max + C_min) / 2
    
    save_image((L * 255).astype(np.uint8), "1.2_Brightness_component.png")
    return H, S, L

# 1.3 Инверсия яркости изображения
def invert_intensity(image):
    _, _, L = rgb_to_hsl(image)
    inverted_image = np.clip(255 - (L * 255), 0, 255).astype(np.uint8)
    save_image(inverted_image, "1.3_Inverted_intensity.png")

# 2.1 Увеличение изображения методом ближайшего соседа
def nearest_neighbor_resize(image, M):
    h, w, c = image.shape
    new_h, new_w = int(h * M), int(w * M)
    resized = np.zeros((new_h, new_w, c), dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            x, y = int(i / M), int(j / M)
            resized[i, j] = image[x, y]
    return resized

# 2.2 Уменьшение изображения методом пропуска строк и столбцов
def downsample(image, N):
    return image[::N, ::N]

# 2.3 Передискретизация в два прохода
def resample_two_pass(image, M, N):
    upsampled = nearest_neighbor_resize(image, M)
    downsampled = downsample(upsampled, N)
    return downsampled

# 2.4 Передискретизация за один проход
def resample_one_pass(image, K):
    return nearest_neighbor_resize(image, K)

# === Получение параметров от пользователя ===
image_path = "image.png"
M = float(input("Введите коэффициент увеличения M: "))
N = int(input("Введите коэффициент уменьшения N: "))
K = M / N  # Автоматический расчет K

image = load_image(image_path)

split_rgb(image)
rgb_to_hsl(image)
invert_intensity(image)

save_image(nearest_neighbor_resize(image, M), "2.1_Interpolation.png")
save_image(downsample(image, N), "2.2_Decimation.png")

resized_MN = resample_two_pass(image, M, N)
save_image(resized_MN, "2.3_Resized_MN.png")

resized_K = resample_one_pass(image, K)
save_image(resized_K, "2.4_Resized_K.png")
