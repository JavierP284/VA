import torch as t
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1. Preprocesamiento: Definimos las transformaciones para normalizar las imágenes
transform = transforms.Compose([
    transforms.ToTensor(),                    # Convierte la imagen a tensor
    transforms.Normalize((0.5,), (0.5,))      # Normaliza los valores de píxel
])

# Cargamos el dataset MNIST para entrenamiento y lo preprocesamos
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = t.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Cargamos el dataset MNIST para prueba y lo preprocesamos
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = t.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# 2. Definición de la red neuronal
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(28*28, 100)  # Capa oculta 1: 784 -> 100
        self.linear2 = nn.Linear(100, 50)     # Capa oculta 2: 100 -> 50
        self.final = nn.Linear(50, 10)        # Capa de salida: 50 -> 10 (dígitos)

    def forward(self, x):
        x = x.view(-1, 28*28)                 # Aplana la imagen a un vector de 784
        x = F.relu(self.linear1(x))           # Act. ReLU en la primera capa
        x = F.relu(self.linear2(x))           # Act. ReLU en la segunda capa
        x = self.final(x)                     # Capa de salida (sin activación)
        return x

# Uso de GPU si está disponible, si no usa CPU
device = t.device("cuda" if t.cuda.is_available() else "cpu")
net = Net().to(device)                        # Mueve la red al dispositivo

# 3. Definimos la función de pérdida y el optimizador
loss_fn = nn.CrossEntropyLoss()               # Pérdida para clasificación multiclase
optimizer = t.optim.Adam(net.parameters(), lr=0.001)  # Optimizador Adam

# 4. Entrenamiento de la red
epochs = 10
for epoch in range(epochs):
    net.train()                              # Modo entrenamiento
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)    # Mueve datos a GPU/CPU
        optimizer.zero_grad()                # Reinicia gradientes
        output = net(x)                      # Forward pass
        loss = loss_fn(output, y)            # Calcula la pérdida
        loss.backward()                      # Backpropagation
        optimizer.step()                     # Actualiza los pesos
        total_loss += loss.item()            # Acumula la pérdida
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# 5. Evaluación eficiente en el set de prueba
net.eval()                                   # Modo evaluación
correct = 0
total = 0
with t.no_grad():                            # No calcula gradientes
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = net(x)
        pred = output.argmax(dim=1)          # Predicción: clase con mayor probabilidad
        correct += (pred == y).sum().item()  # Cuenta aciertos
        total += y.size(0)                   # Cuenta total de muestras

print(f'\nAccuracy: {round(correct / total, 3)}')  # Imprime la precisión

# 6. Mostrar una imagen de test y su predicción
sample_img, sample_label = next(iter(test_loader))  # Toma un batch del set de prueba
sample_img = sample_img[3].to(device)               # Selecciona la imagen 4 del batch
plt.imshow(sample_img.cpu().view(28, 28), cmap='gray')  # Muestra la imagen
plt.title('Imagen de prueba')
plt.axis('off')
plt.show()

prediccion = t.argmax(net(sample_img.view(-1, 784))[0]).item()  # Predice la clase
print(f'Predicción del modelo para esta imagen: {prediccion}')

