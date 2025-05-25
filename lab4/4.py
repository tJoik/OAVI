# Подключаем библиотеки
import cv2
import numpy as np

# Загрузка изображения PNG (ваше исходное цветное)
image = cv2.imread('image.png')

# Приводим в полутоновое изображение
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.png', gray) # сохраняем полутоновое изображение

# Оператор Прюитта (твой вариант 7)
kernelx = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]], dtype=int)

kernely = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]], dtype=int)

# Градиенты по X и Y
Gx = cv2.filter2D(gray, cv2.CV_32F, kernelx)
Gy = cv2.filter2D(gray, cv2.CV_32F, kernely)

# Итоговый градиент G
G = np.sqrt(Gx**2 + Gy**2)

# Нормализуем Gx, Gy, и G в диапазон [0,255]
Gx_norm = cv2.normalize(Gx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
Gy_norm = cv2.normalize(Gy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
G_norm = cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Сохраняем нормализованные фото
cv2.imwrite('Gradient_X.png', Gx_norm)
cv2.imwrite('Gradient_Y.png', Gy_norm)
cv2.imwrite('Gradient_Result.png', G_norm)

# Подбираем порог бинаризации (примерное значение, подбирается опытным путем)
# Например 100 (можешь менять, подбирая самостоятельно!)
_, binary_G = cv2.threshold(G_norm, 100, 255, cv2.THRESH_BINARY)
cv2.imwrite('Gradient_Binary.png', binary_G)

# Покажем все изображения для контроля (это удобно для подбора порога!)
cv2.imshow('Gray Image', gray)
cv2.imshow('Gradient X', Gx_norm)
cv2.imshow('Gradient Y', Gy_norm)
cv2.imshow('Gradient Result', G_norm)
cv2.imshow('Gradient Binary', binary_G)

cv2.waitKey(0)
cv2.destroyAllWindows()
