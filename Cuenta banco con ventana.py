import tkinter as tk
from tkinter import messagebox

class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
        messagebox.showinfo("Cuenta Creada", f"Cuenta bancaria creada para {self.owner}.\nSaldo inicial: ${self.balance}")
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            messagebox.showinfo("Depósito", f"Depósito de ${amount} realizado.\nNuevo saldo: ${self.balance}")
        else:
            messagebox.showwarning("Error", "El monto del depósito debe ser positivo.")
    
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            messagebox.showinfo("Retiro", f"Retiro de ${amount} realizado.\nNuevo saldo: ${self.balance}")
        elif amount > self.balance:
            messagebox.showwarning("Error", f"Retiro de ${amount}. Fondos insuficientes.")
        else:
            messagebox.showwarning("Error", "El monto del retiro debe ser positivo.")
    
    def get_balance(self):
        messagebox.showinfo("Saldo Actual", f"Saldo final: ${self.balance}")
        return self.balance

# Creación de una cuenta bancaria
author_account = BankAccount("Javier Perez", 1000)

# Demostración de los métodos
author_account.deposit(500)
author_account.withdraw(200)
author_account.withdraw(2000)  # Intento de retiro sin saldo suficiente

# Mostrar saldo final
author_account.get_balance()

# Mantener la ventana abierta
root = tk.Tk()
root.withdraw()
root.mainloop()
