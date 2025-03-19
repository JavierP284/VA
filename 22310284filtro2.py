import numpy as np
import cv2
import matplotlib.pyplot as plt


imagen = cv2.imread('Cameraman.png', cv2.IMREAD_GRAYSCALE)

# Duplicar el borde
borde_duplicado = cv2.copyMakeBorder(imagen,1,1,1,1,cv2.BORDER_REPLICATE)
kernel_personalizado = np.array([[1, 1, 1],
                                 [1, -8, 1],
                                 [1, 1, 1]])

# Aplicar el filtro personalizado
imagen_filtrada2 = cv2.filter2D(borde_duplicado,-1, kernel_personalizado)


# Recortar la imagen filtrada para que coincida con el tamaño original
imagen_filtrada2_recortada = imagen_filtrada2[1:-1, 1:-1]

# Guardar en formato .npy
np.save('22310284filtro2.npy', np.array([imagen, imagen_filtrada2_recortada]))

plt.figure(figsize=(10, 6))

# Imagen original
plt.subplot(1, 3, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

# Imagen con filtro Personalizado
plt.subplot(1, 3, 2)
plt.imshow(imagen_filtrada2_recortada, cmap='gray')
plt.title('Filtro Personalizado')
plt.axis('off')

plt.show()