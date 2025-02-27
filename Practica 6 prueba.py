import cv2
import numpy as np

# Cargar imagen en escala de grises
imagen = cv2.imread("ladrillo.jpg", cv2.IMREAD_GRAYSCALE)

# Verificar que la imagen se haya cargado correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen.")
    exit()

# Definir kernel
kernel = np.array([[1, 1, 1], [-1, -1, -1], [0, 0, 0]], dtype=np.int32)

# Obtener dimensiones de la imagen
m, n = imagen.shape

# Crear imagen de salida en ceros (misma dimensión que la imagen original)
imagenf = np.zeros((m, n), dtype=np.uint8)

# Aplicar la convolución manualmente
for x in range(m - 2):
    for y in range(n - 2):
        region = imagen[x:x+3, y:y+3]  # Extraer región de 3x3
        res = np.sum(region * kernel)  # Convolución (multiplicación y suma)
        
        # Aplicar umbral
        if res > 50:
            imagenf[x, y] = 255  # Blanco
        else:
            imagenf[x, y] = 0  # Negro

# Mostrar imagen filtrada
cv2.imshow("Ladrillo Filtrado", imagenf)
cv2.waitKey(0)
cv2.destroyAllWindows()
