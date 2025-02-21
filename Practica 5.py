import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def clasificador(imagen):
    """Clasifica los píxeles según un rango de color y crea una imagen binaria."""
    mascara = (15 < imagen[:, :, 0]) & (imagen[:, :, 0] < 236) & \
              (44 < imagen[:, :, 1]) & (imagen[:, :, 1] < 249) & \
              (100 < imagen[:, :, 2]) & (imagen[:, :, 2] < 254)
    return (mascara * 255).astype(np.uint8)

def etiquetar_objetos(imagen_binaria):
    """Etiqueta objetos, asigna un color único y muestra el número en cada región."""
    etiquetas = label(imagen_binaria)
    num_objetos = etiquetas.max()
    
    # Crear imagen en color con etiquetas únicas
    imagen_etiquetada = np.zeros((*etiquetas.shape, 3), dtype=np.uint8)
    colores = np.random.randint(0, 255, size=(num_objetos + 1, 3))  # Colores aleatorios
    
    for i in range(1, num_objetos + 1):
        imagen_etiquetada[etiquetas == i] = colores[i]

    # Agregar números en cada región detectada
    imagen_etiquetada_con_numeros = imagen_etiquetada.copy()
    regiones = regionprops(etiquetas)
    
    for i, region in enumerate(regiones):
        minr, minc, maxr, maxc = region.bbox
        centro = (minc + maxc) // 2, (minr + maxr) // 2
        cv2.putText(imagen_etiquetada_con_numeros, str(i + 1), centro, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return etiquetas, imagen_etiquetada, imagen_etiquetada_con_numeros, num_objetos

def localizar_objetos(imagen_etiquetada, etiquetas):
    """Dibuja rectángulos alrededor de los objetos sin modificar los colores de la imagen etiquetada."""
    imagen_localizacion = imagen_etiquetada.copy()
    regiones = regionprops(etiquetas)

    for region in regiones:
        minr, minc, maxr, maxc = region.bbox
        cv2.rectangle(imagen_localizacion, (minc, minr), (maxc, maxr), (255, 255, 0), 2)

    return imagen_localizacion

# Cargar imagen y convertir a RGB
imagen = cv2.imread("senas.jpg")
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Paso 1: Clasificación
imagen_binaria = clasificador(imagen)

# Paso 2: Etiquetado con números
etiquetas, imagen_etiquetada, imagen_etiquetada_con_numeros, num_objetos = etiquetar_objetos(imagen_binaria)

# Paso 3: Localización sobre la imagen etiquetada
imagen_localizada = localizar_objetos(imagen_etiquetada, etiquetas)

# Mostrar resultados
fig, axes = plt.subplots(4, 1, figsize=(8, 12))
axes[0].imshow(imagen_rgb)
axes[0].set_title("Imagen Entrada")

axes[1].imshow(imagen_binaria, cmap="gray")
axes[1].set_title("Clasificación (Binaria)")

axes[2].imshow(imagen_etiquetada_con_numeros)
axes[2].set_title(f"Etiquetaje con Números ({num_objetos} objetos detectados)")

axes[3].imshow(imagen_localizada)
axes[3].set_title("Localización de Objetos (Colores + Rectángulos)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
