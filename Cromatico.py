import cv2
import numpy as np

# Cargar la imagen
image_path = "caballo.jpg"
image = cv2.imread(image_path)

# Verificar si la imagen se cargó correctamente
if image is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta.")
    exit()

# Crear imágenes con diferente intensidad
image_low_intensity = np.clip(image * 0.7, 0, 255).astype(np.uint8)
image_high_intensity = np.clip(image * 0.3, 0, 255).astype(np.uint8)

# Función para convertir a coordenadas cromáticas
def to_chromatic_coordinates(img):
    img = img.astype(np.float32)
    sum_channels = np.sum(img, axis=2, keepdims=True) + 1e-6  # Evita división por cero
    return img / sum_channels

# Convertir las imágenes a cromáticas
image_chromatic = to_chromatic_coordinates(image)
image_low_chromatic = to_chromatic_coordinates(image_low_intensity)
image_high_chromatic = to_chromatic_coordinates(image_high_intensity)

# Función de clasificación (segmentación en canal rojo)
def classify_image(img):
    red_channel = img[:, :, 0]
    mask = red_channel > 0.4  # Umbral ajustable
    return (mask * 255).astype(np.uint8)

# Aplicar clasificador
classified_original = classify_image(image_chromatic)
classified_low = classify_image(image_low_chromatic)
classified_high = classify_image(image_high_chromatic)

# Mostrar imágenes en ventanas nuevas (sin convertir a RGB)
cv2.imshow("Imagen Original", image)  # La imagen original debe estar en formato BGR
cv2.imshow("Baja Intensidad", image_low_intensity)
cv2.imshow("Alta Intensidad", image_high_intensity)
cv2.imshow("Imagen Cromática Original", image_chromatic)
cv2.imshow("Imagen Cromática Baja Intensidad", image_low_chromatic)
cv2.imshow("Imagen Cromática Alta Intensidad", image_high_chromatic)
cv2.imshow("Clasificación Original", classified_original)
cv2.imshow("Clasificación Baja Intensidad", classified_low)
cv2.imshow("Clasificación Alta Intensidad", classified_high)

# Esperar hasta que se presione una tecla y luego cerrar las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()
