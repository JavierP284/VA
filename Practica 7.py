#Javier Alejandro Pérez Nuño 22310284 6E2
import cv2
import numpy as np

# Leer la imagen en escala de grises
image = cv2.imread('filtrar.jpeg', cv2.IMREAD_GRAYSCALE)
#image = cv2.imread('filtrar.jpeg')

# Verificar si la imagen se cargó correctamente
if image is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta del archivo.")
    exit()

# Contaminar la imagen con ruido gaussiano
mean = 0
var = 100
sigma = var ** 0.5
gaussian_noise = np.random.normal(mean, sigma, image.shape).reshape(image.shape)
noisy_gaussian = image + gaussian_noise
noisy_gaussian = np.clip(noisy_gaussian, 0, 255).astype(np.uint8)

# Contaminar la imagen con ruido sal y pimienta
s_vs_p = 0.5
amount = 0.04
noisy_sp = np.copy(image)

# Sal
num_salt = np.ceil(amount * image.size * s_vs_p)
coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
noisy_sp[tuple(coords)] = 255

# Pimienta
num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
noisy_sp[tuple(coords)] = 0

# Aplicar filtros a la imagen con ruido gaussiano
filtered_gaussian_gaussian = cv2.GaussianBlur(noisy_gaussian, (5, 5), 0)
filtered_gaussian_mean = cv2.blur(noisy_gaussian, (5, 5))
filtered_gaussian_median = cv2.medianBlur(noisy_gaussian, 5)
filtered_gaussian_min = cv2.erode(noisy_gaussian, np.ones((5, 5), np.uint8))
filtered_gaussian_max = cv2.dilate(noisy_gaussian, np.ones((5, 5), np.uint8))

# Aplicar filtros a la imagen con ruido sal y pimienta
filtered_sp_gaussian = cv2.GaussianBlur(noisy_sp, (5, 5), 0)
filtered_sp_mean = cv2.blur(noisy_sp, (5, 5))
filtered_sp_median = cv2.medianBlur(noisy_sp, 5)
filtered_sp_min = cv2.erode(noisy_sp, np.ones((5, 5), np.uint8))
filtered_sp_max = cv2.dilate(noisy_sp, np.ones((5, 5), np.uint8))

# Mostrar las imágenes resultantes usando cv2
cv2.imshow('Original', image)
cv2.imshow('Ruido Gaussiano', noisy_gaussian)
cv2.imshow('Ruido Sal y Pimienta', noisy_sp)

cv2.imshow('Gaussiano (Gaussiano)', filtered_gaussian_gaussian)
cv2.imshow('Media (Gaussiano)', filtered_gaussian_mean)
cv2.imshow('Mediana (Gaussiano)', filtered_gaussian_median)
cv2.imshow('Minimo (Gaussiano)', filtered_gaussian_min)
cv2.imshow('Maximo (Gaussiano)', filtered_gaussian_max)

cv2.imshow('Gaussiano (Sal y Pimienta)', filtered_sp_gaussian)
cv2.imshow('Media (Sal y Pimienta)', filtered_sp_mean)
cv2.imshow('Mediana (Sal y Pimienta)', filtered_sp_median)
cv2.imshow('Minimo (Sal y Pimienta)', filtered_sp_min)
cv2.imshow('Maximo (Sal y Pimienta)', filtered_sp_max)

# Esperar a que el usuario presione una tecla y cerrar las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()