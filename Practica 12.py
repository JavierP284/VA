# Javier Alejandro Pérez Nuño 22310284 6E2
import cv2
import numpy as np

# Ruta del video (asegúrate de definirla correctamente)
vid_path = "ruta_del_video.mp4"  # Cambia esto por la ruta de tu video

# Inicialización de la captura de video y el sustractor de fondo
cap = cv2.VideoCapture(vid_path)
backSub = cv2.createBackgroundSubtractorMOG2()

if not cap.isOpened():
    print("Error al abrir el archivo de video")
else:
    while cap.isOpened():
        # Captura frame por frame
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o error al leer el frame")
            break

        # Aplicar sustracción de fondo
        fg_mask = backSub.apply(frame)

        # Aplicar umbral global para eliminar sombras
        retval, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

        # Definir el kernel y aplicar erosión
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

        # Encontrar contornos
        contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar contornos grandes
        min_contour_area = 500  # Define tu umbral mínimo de área
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        # Dibujar contornos y rectángulos en el frame original
        frame_out = frame.copy()
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            frame_out = cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 0, 200), 3)

        # Mostrar el frame resultante
        cv2.imshow('Frame_final', frame_out)

        # Salir con la tecla 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()