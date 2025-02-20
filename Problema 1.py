# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 23:21:57 2025

@author: javal
"""

# Definir la lista de libros
favorite_books = [
    "Maze Runner: La Cura Mortal",
    "Maze Runner: Correr o Morir",
    "Dune",
    "Assassins Creed: Renaissance",
    "Codigo Cruel"
]

# Imprimir la lista de libros
def print_books(book_list):
    print("\nLista de libros favoritos:")
    for book in book_list:
        print(f"- {book}")

# Agregar un nuevo libro a la lista
def add_books(book_list):
    while True:
        new_book = input("Ingrese el título de un nuevo libro (o escriba 'salir' para terminar): ") #Establecer la opcion de salir del codigo y solicitar nuevo libro para agregar a la listta
        if new_book.lower() == 'salir':
            print("Saliendo del programa...")
            break
        book_list.append(new_book) #Agregar nuevo libro a la lista
        print(f"\"{new_book}\" ha sido agregado a la lista.")
        print_books(book_list)

# Mostrar libros iniciales y permitir agregar más
print_books(favorite_books)
add_books(favorite_books)
