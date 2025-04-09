import numpy as np
import matplotlib.pyplot as plt

x=np.arange(-5, 6)
y=3*x
w=5
y1=w*x
e=(np.sum(np.power(y1-y,2)))
E=e/(2* x.shape[0])

print(f"Error cuadrático medio: {E}")

plt.plot(x, y, label='y=3x')
plt.plot(x, y1, label='y=5x')
plt.title('Regresión Lineal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()


w_values = np.arange(-5,12)
E_values = []
for w in w_values:
    y1 = w * x
    e = np.sum(np.power(y1 - y, 2))
    E = e / (2 * x.shape[0])
    E_values.append(E)

# Graficar E en función de w
plt.plot(w_values, E_values)
plt.title('Error cuadrático medio en función de w')
plt.xlabel('w')
plt.ylabel('Error cuadrático medio (E)')
plt.grid()
plt.show()