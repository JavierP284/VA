# Javier Alejandro Pérez Nuño 22310284 6E2
import cv2

# Cargar la imagen
imagen = cv2.imread('Tornillos.jpg')

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen 'Tornillos.jpg'. Verifica la ruta del archivo.")
    exit()

# Convertir a escala de grises
grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Detectar bordes
bordes = cv2.Canny(grises, 100, 200)

# Encontrar contornos (para OpenCV 4)
ctns, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar contornos
cv2.drawContours(imagen, ctns, -1, (0, 0, 255), 2)

# Mostrar el número de contornos encontrados
print('Número de contornos encontrados: ', len(ctns))
texto = 'Contornos encontrados: ' + str(len(ctns))

# Agregar texto a la imagen
cv2.putText(imagen, texto, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

# Mostrar la imagen
cv2.imshow('Imagen', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()