import numpy as np
import cv2
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
imagen = cv2.imread('Cameraman.png', cv2.IMREAD_GRAYSCALE)

# Duplicar el borde
borde_duplicado = cv2.copyMakeBorder(imagen, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

# Definir el kernel del filtro Laplaciano
kernel_laplaciano = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])

# Aplicar filtro Laplaciano con el borde duplicado
imagen_laplaciana = cv2.filter2D(borde_duplicado, -1, kernel_laplaciano)

# Guardar ambas imágenes (con y sin recorte) como un diccionario en el archivo .npy
np.save('22310284filtro1.npy', {'imagen_original': imagen, 'imagen_laplaciana': imagen_laplaciana})

# Mostrar las imágenes
plt.figure(figsize=(10, 6))

# Imagen original
plt.subplot(1, 3, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

# Imagen con filtro Laplaciano (con borde duplicado)
plt.subplot(1, 3, 2)
plt.imshow(imagen_laplaciana, cmap='gray')
plt.title('Filtro Laplaciano con Borde Duplicado')
plt.axis('off')

plt.show()
