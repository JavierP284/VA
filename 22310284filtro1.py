import numpy as np
import cv2
import matplotlib.pyplot as plt


imagen = cv2.imread('Cameraman.png', cv2.IMREAD_GRAYSCALE)

# Duplicar el borde
borde_duplicado = cv2.copyMakeBorder(imagen,1,1,1,1,cv2.BORDER_REPLICATE)

kernel_laplaciano = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])

# Aplicar filtro Laplaciano
imagen_laplaciana = cv2.filter2D(borde_duplicado,-1, kernel_laplaciano)

# Recortar la imagen filtrada para que coincida con el tamaño original
imagen_laplaciana_recortada = imagen_laplaciana[1:-1, 1:-1]

# Guardar en formato .npy
np.save('22310284filtro1.npy', np.array([borde_duplicado, imagen_laplaciana]))

plt.figure(figsize=(10, 6))

# Imagen original
plt.subplot(1, 3, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

# Imagen con filtro Laplaciano
plt.subplot(1, 3, 2)
plt.imshow(imagen_laplaciana_recortada, cmap='gray')
plt.title('Filtro Laplaciano')
plt.axis('off')

plt.show()
