#Javier Alejandro Pérez Nuño 22310284 6E2
"""
Created on Wed Feb 12 23:21:57 2025

@author: javal
"""

# Definir una tupla con tres países
countries_to_visit = ("Japon", "Italia", "Canada")

# Imprimir cada país en la tupla
def print_countries():
    print("\nPaíses que me gustaría visitar:")
    for country in countries_to_visit:
        print(country)

# Verificar si un país está en la tupla
def check_country():
    while True:
        country = input("Escribe el nombre de un país para verificar si está en la lista (o escribe 'salir' para terminar): ") #Establecer la opcion de salir
        if country.lower() == 'salir':
            print("Saliendo del programa...")
            break
        if country in countries_to_visit:
            print(f"{country} está en la lista de países que quiero visitar.")
        else:
            print(f"{country} no está en la lista de países que quiero visitar.")

#Imprimir Paises y si estan o no en la lista
print_countries()
check_country()
