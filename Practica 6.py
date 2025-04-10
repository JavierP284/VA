#Javier Alejandro Pérez Nuño 22310284 6E2
import cv2

orig = cv2.imread("Perro_salchicha.jpg")
image = orig.copy()
template = cv2.imread("Muestra.jpg")


image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
methods = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]

for method in methods:
    res = cv2.matchTemplate(image_gray, template_gray, method=method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(min_val, max_val, min_loc, max_loc)

    if method == cv2.TM_SQDIFF or method == cv2.TM_SQDIFF_NORMED:
        x1, y1 = min_loc
        x2, y2 = min_loc[0] + template.shape[1], min_loc[1] + template.shape[0] #Localizacion del valor minimo
    else:
        x1, y1 = max_loc
        x2, y2 = max_loc[0] + template.shape[1], max_loc[1] + template.shape[0] #En todas las demas localizacion del valor maximo

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imshow("Image", image)
    cv2.imshow("Template", template)
    image = orig.copy()
    cv2.waitKey(0)
cv2.destroyAllWindows()