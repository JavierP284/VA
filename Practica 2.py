# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:25:15 2025

@author: javal
"""

import cv2
import numpy as np
imagen=cv2.imread("caballo.jpg")

def ajustar_brillo(imagen, factor):
    return np.clip(imagen * factor, 0, 255).astype(np.uint8)

def to_chromatic_coordinates(img):
    img = img.astype(np.float32)
    sum_channels = np.sum(img, axis=2, keepdims=True) + 1e-6  # Evita división por cero
    return img / sum_channels

def clasificador(imagen):
    m,n,c=imagen.shape
    imagenb=np.zeros((m,n))
    imageng=np.zeros((m,n))
    imagenr=np.zeros((m,n))
    
    for x in range (m):
        for y in range(n):
                #if 25<imagen[x,y,0]<86 and 48<imagen[x,y,1]<131 and 44<imagen[x,y,2]<193:
                
                    rgb=imagen[x,y,0]+imagen[x,y,1]+imagen[x,y,2]
                    imagenb[x,y]=(imagen[x,y,0])/rgb
                    imageng[x,y]=(imagen[x,y,1])/rgb
                    imagenr[x,y]=(imagen[x,y,2])/rgb
                
    # Clasificación en el canal rojo
    mask = imagenr > 0.4  # Umbral ajustable
    return (mask * 255).astype(np.uint8)
            
    
imagen_1= ajustar_brillo(imagen, 0.7) 
imagen_2= ajustar_brillo(imagen, 0.3) 

imagen_crom1= to_chromatic_coordinates(imagen)
imagen_crom2= to_chromatic_coordinates(imagen_1)
imagen_crom3= to_chromatic_coordinates(imagen_2)

# Aplicar el clasificador a las imágenes cromáticas
imagen_clas1 = clasificador(imagen_crom1)
imagen_clas2 = clasificador(imagen_crom2)
imagen_clas3 = clasificador(imagen_crom3)

#Mostrar Imagenes
cv2.imshow("Imagen Original", imagen)
cv2.imshow("Imagen 0.7", imagen_1)
cv2.imshow("Imagen 0.3", imagen_2)
cv2.imshow("Imagen Cromatica Original", imagen_crom1)
cv2.imshow("Imagen Cromatica 0.7", imagen_crom2)
cv2.imshow("Imagen Cromatica 0.3", imagen_crom3)
cv2.imshow("Imagen Clasificacion Original", imagen_clas1)
cv2.imshow("Imagen Clasificacion 0.7", imagen_clas2)
cv2.imshow("Imagen Clasificacion 0.1", imagen_clas3)

cv2.waitKey(0)
cv2.destroyAllWindows()

#b->0
#g->1
#r->2
#rgb=167 r/rgb=0.18 g/rgb