# Javier Alejandro Pérez Nuño 22310284 6E2
import cv2
import numpy as np

# Inicializar la captura de video desde un archivo
cap = cv2.VideoCapture('Personas caminando.mp4')

# Crear un sustractor de fondo (Background Subtractor) utilizando MOG2
backSub = cv2.createBackgroundSubtractorMOG2()

# Verificar si el archivo de video se abrió correctamente
if not cap.isOpened():
    print("Error opening video file")  # Mostrar mensaje de error si no se puede abrir
    exit()

# Obtener las propiedades del video original
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ancho del video
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Alto del video
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Fotogramas por segundo

# Inicializar el objeto VideoWriter para guardar el video procesado
output_file = 'Personas_caminando_procesado.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para el archivo de salida
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Procesar cada fotograma del video en un bucle
while cap.isOpened():
    # Leer el siguiente fotograma del video
    ret, frame = cap.read()
    if not ret:  # Si no se pudo leer el fotograma (fin del video o error)
        print("No se pudo leer el fotograma. Finalizando...")
        break

    # Aplicar la sustracción de fondo para obtener la máscara de movimiento
    fg_mask = backSub.apply(frame)

    # Reducir el ruido en la máscara utilizando operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Crear un kernel elíptico
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Cerrar agujeros en las regiones detectadas
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)   # Eliminar ruido pequeño

    # Encontrar los contornos en la máscara procesada
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar los contornos para quedarse solo con los suficientemente grandes
    min_contour_area = 1000  # Área mínima para considerar un contorno como válido
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Dibujar rectángulos alrededor de los objetos detectados
    frame_out = frame.copy()  # Crear una copia del fotograma original para dibujar los rectángulos
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)  # Obtener las coordenadas del rectángulo delimitador
        cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dibujar el rectángulo en verde

    # Guardar el fotograma procesado en el archivo de salida
    out.write(frame_out)

    # Mostrar el fotograma procesado con los rectángulos dibujados
    cv2.imshow('Frame_final', frame_out)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Liberar los recursos utilizados
cap.release()  # Liberar el archivo de video
out.release()  # Liberar el archivo de salida
cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV