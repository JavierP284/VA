import cv2
import numpy as np


imagen=cv2.imread("ladrillo.jpg")
kernel=np.array([[1,1,1],[-1,-1,-1],[0,0,0]])
m,n=imagen.shape
imagenf=np.zeros_like(imagen)


for x in range(m-2):
    for y in range(n-2):
        res=np.sum(imagen[x:x+3,y:y+3]*kernel)
        if res>50:
            imagenf[x,y]=255


imagenf=imagenf.astype(np.uint8)
cv2.imshow("ladrillo",imagenf)
cv2.waitKey(0)
cv2.destroyAllWindows()