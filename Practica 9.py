#Javier Alejandro Pérez Nuño 22310284 6E2
import cv2
import numpy as np

# Leer la imagen original
img = cv2.imread('ladrillo.jpg') 
# Verificar si la imagen se ha cargado correctamente
if img is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta.")
    exit()

# Mostrar imagen original
cv2.imshow('Original', img)
cv2.waitKey(0)

# Convertir a escala de grises
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar desenfoque gaussiano para mejor detección de bordes
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

# Detección de bordes con Sobel
# Sobel en dirección X (detecta bordes verticales)
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
# Sobel en dirección Y (detecta bordes horizontales)
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)

# Calcular la magnitud del gradiente (combinación de X e Y)
sobel_mag = np.sqrt(sobelx**2 + sobely**2)
sobel_mag = np.uint8(sobel_mag / sobel_mag.max() * 255)

# Calcular la dirección del gradiente
sobel_dir = np.arctan2(sobely, sobelx)

# Mostrar resultados de Sobel
cv2.imshow('Sobel X', cv2.convertScaleAbs(sobelx))
cv2.waitKey(0)
cv2.imshow('Sobel Y', cv2.convertScaleAbs(sobely))
cv2.waitKey(0)
cv2.imshow('Sobel Magnitud', sobel_mag)
cv2.waitKey(0)

# Opcional: Mostrar dirección del gradiente (normalizada para visualización)
sobel_dir_vis = np.uint8((sobel_dir + np.pi) * (255 / (2 * np.pi)))
cv2.imshow('Sobel Direccion', sobel_dir_vis)
cv2.waitKey(0)

# Combinación de X e Y usando la función Sobel (no recomendado)
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
cv2.imshow('Sobel X Y usando Sobel() function', cv2.convertScaleAbs(sobelxy))
cv2.waitKey(0)

# Detección de bordes con Canny para comparación
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

cv2.destroyAllWindows()