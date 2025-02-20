"""
Created on Wed Feb 12 23:21:57 2025

@author: javal
"""

#Crear la cuenta bancaria en una clase
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        print(f"Cuenta Bancaria creada para {self.owner}. Saldo Inicial: ${self.balance}")
        
#Crear primer deposito  
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            print(f"Depósito de ${amount} realizado. Nuevo saldo: ${self.balance}")
        else:
            print("El monto del depósito debe ser positivo.")
            
#Crear primer retiro   
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            print(f"Retiro de ${amount} realizado. Nuevo saldo: ${self.balance}")
        elif amount > self.balance:
            print(f"Retiro de ${amount} no permitido. Fondos insuficientes.")  #Caso de retiro cuando no se ttiene saldo suficiente
        else:
            print("El monto del retiro debe ser positivo.")

#Generar saldo actual
    def get_balance(self):
        return self.balance

# Creación de una cuenta bancaria
author_account = BankAccount("Javier Perez", 1000)

# Demostración de los métodos
author_account.deposit(500)
author_account.withdraw(200)
author_account.withdraw(2000)  # Intento de retiro sin saldo suficiente

# Mostrar saldo final
print(f"Saldo final: ${author_account.get_balance()}")
