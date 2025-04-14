# Javier Alejandro Pérez Nuño 22310284 6E2
import cv2
import numpy as np

# Crear las imágenes base
img1 = np.zeros((400, 600), dtype=np.uint8)
# Coordenadas del hexágono
hexagon_points = np.array([[300, 100], [400, 175], [400, 275], [300, 350], [200, 275], [200, 175]], np.int32)
hexagon_points = hexagon_points.reshape((-1, 1, 2))
cv2.fillPoly(img1, [hexagon_points], 255)

img2 = np.zeros((400, 600), dtype=np.uint8)
# Coordenadas del triángulo
triangle_points = np.array([[300, 100], [400, 300], [200, 300]], np.int32)
triangle_points = triangle_points.reshape((-1, 1, 2))
cv2.fillPoly(img2, [triangle_points], 255)

# Mostrar las imágenes originales
cv2.imshow('img1 (Hexagono)', img1)
cv2.imshow('img2 (Triangulo)', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Operación AND
AND = cv2.bitwise_and(img1, img2)
cv2.imshow('AND', AND)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Operación NOT
NOT = cv2.bitwise_not(img1)
cv2.imshow('NOT', NOT)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Operación OR
OR = cv2.bitwise_or(img1, img2)
cv2.imshow('OR', OR)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Operación XOR
XOR = cv2.bitwise_xor(img1, img2)
cv2.imshow('XOR', XOR)
cv2.waitKey(0)
cv2.destroyAllWindows()