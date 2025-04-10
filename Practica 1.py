#Javier Alejandro Pérez Nuño 22310284 6E2
"""
Created on Wed Feb 12 16:25:15 2025

@author: javal
"""

import cv2
import numpy as np
imagen=cv2.imread("caballo.jpg")
m,n,c=imagen.shape
imagenb=np.zeros((m,n))

for x in range (m):
    for y in range(n):
        if 25<imagen[x,y,0]<86 and 48<imagen[x,y,1]<131 and 44<imagen[x,y,2]<193:
            imagenb[x,y]=255
            
cv2.imshow("caballo",imagenb)
cv2.waitKey(0)
cv2.destroyAllWindows()