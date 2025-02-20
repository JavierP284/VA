import cv2
import numpy as np

def organizar_imagenes(imagenes, filas, columnas, escala=0.3):
    h, w = int(imagenes[0].shape[0] * escala), int(imagenes[0].shape[1] * escala)
    imagenes_redimensionadas = [cv2.resize(img, (w, h)) for img in imagenes]
    
    imagen_grilla = np.zeros((h * filas, w * columnas, 3), dtype=np.uint8)
    
    for idx, img in enumerate(imagenes_redimensionadas):
        if idx >= filas * columnas:
            break  # Evita exceder la cuadrícula
        fila = idx // columnas
        columna = idx % columnas
        imagen_grilla[fila * h:(fila + 1) * h, columna * w:(columna + 1) * w] = img
    
    return imagen_grilla

# Cargar imagen
imagen = cv2.imread("caballo.jpg").astype(np.float32)

def cambiar_tono(imagen, canal):
    imagen_modificada = imagen.copy()
    imagen_modificada[:, :, canal] = np.clip(imagen_modificada[:, :, canal] * 1.5, 0, 255)
    return imagen_modificada.astype(np.uint8)

def to_chromatic_coordinates(img):
    img = img.astype(np.float32)
    sum_channels = np.sum(img, axis=2, keepdims=True) + 1e-6
    return img / sum_channels

def white_patch(imagen):
    max_values = np.max(imagen, axis=(0, 1))
    imagen_white_patch = (imagen / max_values) * 255
    return np.clip(imagen_white_patch, 0, 255).astype(np.uint8)

def clasificador(imagen):
    m, n, c = imagen.shape
    imagenr = np.zeros((m, n))
    
    for x in range(m):
        for y in range(n):
            rgb = imagen[x, y, 0] + imagen[x, y, 1] + imagen[x, y, 2]
            if rgb > 0:
                imagenr[x, y] = imagen[x, y, 2] / rgb
    
    mask = imagenr > 0.4
    return (mask * 255).astype(np.uint8)

# Aplicar White Patch antes de modificar los tonos
imagen_wp1 = white_patch(imagen)

# Crear imágenes modificadas
imagen_r = cambiar_tono(imagen_wp1, 2)
imagen_g = cambiar_tono(imagen_wp1, 1)
imagen_b = cambiar_tono(imagen_wp1, 0)

# Aplicar coordenadas cromáticas
imagen_crom1 = to_chromatic_coordinates(imagen_wp1)
imagen_crom2 = to_chromatic_coordinates(imagen_r)
imagen_crom3 = to_chromatic_coordinates(imagen_g)
imagen_crom4 = to_chromatic_coordinates(imagen_b)

# Aplicar clasificador antes de White Patch
imagen_clas1 = clasificador(imagen_crom1)
imagen_clas2 = clasificador(imagen_crom2)
imagen_clas3 = clasificador(imagen_crom3)
imagen_clas4 = clasificador(imagen_crom4)

# Aplicar clasificador después de White Patch
imagen_clas_wp1 = clasificador(to_chromatic_coordinates(imagen_wp1))
imagen_clas_wp2 = clasificador(to_chromatic_coordinates(imagen_r))
imagen_clas_wp3 = clasificador(to_chromatic_coordinates(imagen_g))
imagen_clas_wp4 = clasificador(to_chromatic_coordinates(imagen_b))

# Convertir imágenes en escala de grises a 3 canales para visualización uniforme
imagenes = [imagen_wp1, imagen_r, imagen_g, imagen_b, imagen_crom1, imagen_crom2, imagen_crom3, imagen_crom4,
            imagen_clas1, imagen_clas2, imagen_clas3, imagen_clas4, imagen_clas_wp1, imagen_clas_wp2, imagen_clas_wp3, imagen_clas_wp4]
imagenes = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img for img in imagenes]

# Crear la imagen en cuadrícula con reducción de tamaño
grid = organizar_imagenes(imagenes, 4, 4, escala=0.3)

# Mostrar imagen en una sola ventana
cv2.imshow("Comparación de imágenes", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
