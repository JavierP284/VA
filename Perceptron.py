import numpy as np

w = np.array([1, 1, 1])

X = np.array([[1, -1, -1],
              [1, -1, 1],
              [1, 1, -1],
              [1, 1, 1]
              ])

yd = np.array([-1, -1, -1, 1])

alpha = 0.5

for x, yd_i in zip(X, yd):  # Cambiar el nombre de la variable de iteración
    y = np.sign(np.dot(w, x))  # Producto punto entre w y x
    w = w + alpha * (yd_i - y) * x  # Actualización de w
    print("Esta entrada ", w)

ys = []  # Inicializar la lista para guardar las salidas

for x, yd_i in zip(X, yd):  # Cambiar el nombre de la variable de iteración
    y = np.sign(np.dot(w, x))  # Producto punto entre w y x
    ys.append(y)  # Guardar la salida

ys = np.array(ys)  # Convertir la lista a un array de numpy
print("Salida final: ", ys)  # Imprimir la lista de salidas