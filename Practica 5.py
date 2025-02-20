import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import label2rgb

def clasificador(imagen):
    mascara = (15 < imagen[:, :, 0]) & (imagen[:, :, 0] < 236) & \
              (44 < imagen[:, :, 1]) & (imagen[:, :, 1] < 249) & \
              (100 < imagen[:, :, 2]) & (imagen[:, :, 2] < 254)
    return (mascara * 255).astype(np.uint8)  # Convertir a uint8
    

def etiquetar_objetos(imagen_binaria):
    etiquetas = label(imagen_binaria)
    imagen_etiquetada = label2rgb(etiquetas, bg_label=0)
    return etiquetas, imagen_etiquetada

def localizar_objetos(imagen_original, etiquetas):
    imagen_localizacion = imagen_original.copy()
    regiones = regionprops(etiquetas)
    
    for i, region in enumerate(regiones):
        minr, minc, maxr, maxc = region.bbox
        cv2.rectangle(imagen_localizacion, (minc, minr), (maxc, maxr), (255, 255, 0), 2)
        cv2.putText(imagen_localizacion, str(i+1), (minc, minr-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return imagen_localizacion

# Cargar la imagen
imagen = cv2.imread("senas.jpg")

# Convertir de BGR a RGB para mostrar correctamente
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Paso 1: Clasificación
imagen_binaria = clasificador(imagen)

# Paso 2: Etiquetado
etiquetas, imagen_etiquetada = etiquetar_objetos(imagen_binaria)

# Paso 3: Localización
imagen_localizada = localizar_objetos(imagen, etiquetas)

# Convertir la imagen localizada a RGB
imagen_localizada_rgb = cv2.cvtColor(imagen_localizada, cv2.COLOR_BGR2RGB)

# Mostrar resultados
fig, axes = plt.subplots(4, 1, figsize=(8, 12))
axes[0].imshow(imagen_rgb)
axes[0].set_title("Imagen Entrada")

axes[1].imshow(imagen_binaria, cmap="gray")
axes[1].set_title("Clasificación")

axes[2].imshow(imagen_etiquetada)
axes[2].set_title("Etiquetaje")

axes[3].imshow(imagen_localizada_rgb)
axes[3].set_title("Localización")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()