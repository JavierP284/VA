import cv2
import numpy as np

# Leer la imagen en escala de grises
original=cv2.imread('ladrillo.jpg')
image = cv2.imread('ladrillo.jpg', cv2.IMREAD_GRAYSCALE)

# Verificar si la imagen se cargó correctamente
if image is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta del archivo.")
    exit()

# Definir los kernels para detectar bordes verticales y horizontales
kernel_vertical = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

kernel_horizontal = np.array([[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]])

# Aplicar los filtros para detectar bordes verticales y horizontales
Gx = cv2.filter2D(image, -1, kernel_vertical)
Gy = cv2.filter2D(image, -1, kernel_horizontal)

# Calcular la magnitud del vector gradiente
magnitud_gradiente = np.sqrt(Gx**2 + Gy**2)

# Convertir la magnitud del gradiente a un tipo de dato válido (uint8)
magnitud_gradiente = np.uint8(magnitud_gradiente)

# Normalizar la magnitud del gradiente para visualización
magnitud_gradiente = cv2.normalize(magnitud_gradiente, None, 0, 255, cv2.NORM_MINMAX)

# Mostrar las imágenes resultantes usando cv2
cv2.imshow("Original", original)
cv2.imshow('Bordes Verticales (Gx)', Gx)
cv2.imshow('Bordes Horizontales (Gy)', Gy)
cv2.imshow('Magnitud del Gradiente (|G|)', magnitud_gradiente)

# Esperar a que el usuario presione una tecla y cerrar las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()