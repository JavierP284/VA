# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:11:24 2025

@author: javier_perez
"""

import cv2
import numpy as np

def cambiar_tono(imagen, canal):
    imagen_modificada = imagen.copy()
    imagen_modificada[:, :, canal] = np.clip(imagen_modificada[:, :, canal] * 1.5, 0, 255)
    return imagen_modificada.astype(np.uint8)

def to_chromatic_coordinates(img):
    img = img.astype(np.float32)
    sum_channels = np.sum(img, axis=2, keepdims=True) + 1e-6  # Evita división por cero
    return img / sum_channels

def white_patch(imgen):
    max_values = np.max(imgen, axis=(0, 1))  # Encuentra el valor máximo de cada canal (R, G, B)
    imagen_white_patch = (255 / max_values) * imagen  # Normaliza por el máximo y escala a 255
    return np.clip(imagen_white_patch, 0, 255).astype(np.uint8)

def clasificador(imagen):
    m, n, c = imagen.shape
    imagenr = np.zeros((m, n))
    
    for x in range(m):
        for y in range(n):
            rgb = imagen[x, y, 0] + imagen[x, y, 1] + imagen[x, y, 2]
            if rgb > 0:  # Evita división por cero
                imagenr[x, y] = imagen[x, y, 2] / rgb  # Canal rojo normalizado
    
    mask = imagenr > 0.4  # Umbral ajustable
    return (mask * 255).astype(np.uint8)

# Cargar imagen
imagen = cv2.imread("caballo.jpg").astype(np.float32)

# Crear imágenes con diferentes tonos
imagen_r = cambiar_tono(imagen, 2)  # Modificar canal Rojo
imagen_g = cambiar_tono(imagen, 1)  # Modificar canal Verde
imagen_b = cambiar_tono(imagen, 0)  # Modificar canal Azul

# Aplicar coordenadas cromáticas
imagen_crom1 = to_chromatic_coordinates(imagen)
imagen_crom2 = to_chromatic_coordinates(imagen_r)
imagen_crom3 = to_chromatic_coordinates(imagen_g)
imagen_crom4 = to_chromatic_coordinates(imagen_b)

# Aplicar clasificador antes de White Patch
imagen_clas1 = clasificador(imagen_crom1)
imagen_clas2 = clasificador(imagen_crom2)
imagen_clas3 = clasificador(imagen_crom3)
imagen_clas4 = clasificador(imagen_crom4)

# Aplicar White Patch
imagen_wp1 = white_patch(imagen)
imagen_wp2 = white_patch(imagen_r)
imagen_wp3 = white_patch(imagen_g)
imagen_wp4 = white_patch(imagen_b)

# Aplicar clasificador después de White Patch
imagen_clas_wp1 = clasificador(to_chromatic_coordinates(imagen_wp1))
imagen_clas_wp2 = clasificador(to_chromatic_coordinates(imagen_wp2))
imagen_clas_wp3 = clasificador(to_chromatic_coordinates(imagen_wp3))
imagen_clas_wp4 = clasificador(to_chromatic_coordinates(imagen_wp4))

# Mostrar imágenes
cv2.imshow("Imagen Original", imagen.astype(np.uint8))
cv2.imshow("Imagen Rojo", imagen_r)
cv2.imshow("Imagen Verde", imagen_g)
cv2.imshow("Imagen Azul", imagen_b)
cv2.imshow("Clasificacion Original", imagen_clas1)
cv2.imshow("Clasificacion Rojo", imagen_clas2)
cv2.imshow("Clasificacion Verde", imagen_clas3)
cv2.imshow("Clasificacion Azul", imagen_clas4)
cv2.imshow("White Patch Original", imagen_wp1)
cv2.imshow("White Patch Rojo", imagen_wp2)
cv2.imshow("White Patch Verde", imagen_wp3)
cv2.imshow("White Patch Azul", imagen_wp4)
cv2.imshow("Clasificacion WP Original", imagen_clas_wp1)
cv2.imshow("Clasificacion WP Rojo", imagen_clas_wp2)
cv2.imshow("Clasificacion WP Verde", imagen_clas_wp3)
cv2.imshow("Clasificacion WP Azul", imagen_clas_wp4)

cv2.waitKey(0)
cv2.destroyAllWindows()
