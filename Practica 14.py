# Javier Alejandro Pérez Nuño 22310284 6E2
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Función para verificar si la imagen se cargó correctamente
def verificar_imagen(img, nombre):
    """
    Verifica si una imagen fue cargada correctamente.
    Si no se cargó, lanza una excepción con un mensaje de error.
    """
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {nombre}")

# Escalado de imagen
img = cv2.imread('caballo.jpg')  # Carga la imagen en color
verificar_imagen(img, 'caballo.jpg')  # Verifica que la imagen se haya cargado correctamente

# Escalado utilizando un factor de escala
res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Escala la imagen al doble de tamaño

# Escalado indicando manualmente el nuevo tamaño deseado
height, width = img.shape[:2]  # Obtiene las dimensiones originales de la imagen
res_manual = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)  # Escala manualmente

# Traslación de imagen
img_gray = cv2.imread('caballo.jpg', 0)  # Carga la imagen en escala de grises
verificar_imagen(img_gray, 'caballo.jpg')  # Verifica que la imagen se haya cargado correctamente
rows, cols = img_gray.shape  # Obtiene las dimensiones de la imagen en escala de grises

# Matriz de transformación para traslación
M = np.float32([[1, 0, 210], [0, 1, 20]])  # Traslada la imagen 210 píxeles a la derecha y 20 píxeles hacia abajo
dst_translation = cv2.warpAffine(img_gray, M, (cols, rows))  # Aplica la transformación de traslación

# Muestra la imagen trasladada
cv2.imshow('Traslacion', dst_translation)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Rotación de imagen
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)  # Matriz de rotación (centro, ángulo, escala)
dst_rotation = cv2.warpAffine(img_gray, M, (cols, rows))  # Aplica la transformación de rotación

# Muestra la imagen rotada
cv2.imshow('Rotacion', dst_rotation)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Transformación afín
img_grid = cv2.imread('sudoku.png')  # Carga la imagen en color
verificar_imagen(img_grid, 'sudoku.png')  # Verifica que la imagen se haya cargado correctamente
rows, cols, ch = img_grid.shape  # Obtiene las dimensiones de la imagen

# Puntos originales y transformados para la transformación afín
pts1 = np.float32([[100, 400], [400, 100], [100, 100]])  # Puntos en la imagen original
pts2 = np.float32([[50, 300], [400, 200], [80, 150]])  # Puntos en la imagen transformada

# Matriz de transformación afín
M = cv2.getAffineTransform(pts1, pts2)  # Calcula la matriz de transformación afín
dst_affine = cv2.warpAffine(img_grid, M, (cols, rows))  # Aplica la transformación afín

# Muestra la imagen original y la transformada
plt.subplot(121), plt.imshow(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB)), plt.title('Input')  # Imagen original
plt.subplot(122), plt.imshow(cv2.cvtColor(dst_affine, cv2.COLOR_BGR2RGB)), plt.title('Output')  # Imagen transformada
plt.show()

# Transformación de perspectiva
img_sudoku = cv2.imread('sudoku.png')  # Carga la imagen en color
verificar_imagen(img_sudoku, 'sudoku.png')  # Verifica que la imagen se haya cargado correctamente
rows, cols, ch = img_sudoku.shape  # Obtiene las dimensiones de la imagen

# Puntos originales y transformados para la transformación de perspectiva
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])  # Puntos en la imagen original
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])  # Puntos en la imagen transformada

# Matriz de transformación de perspectiva
M = cv2.getPerspectiveTransform(pts1, pts2)  # Calcula la matriz de transformación de perspectiva
dst_perspective = cv2.warpPerspective(img_sudoku, M, (300, 300))  # Aplica la transformación de perspectiva

# Muestra la imagen original y la transformada
plt.subplot(121), plt.imshow(cv2.cvtColor(img_sudoku, cv2.COLOR_BGR2RGB)), plt.title('Input')  # Imagen original
plt.subplot(122), plt.imshow(cv2.cvtColor(dst_perspective, cv2.COLOR_BGR2RGB)), plt.title('Output')  # Imagen transformada
plt.show()