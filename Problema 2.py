#Javier Alejandro Pérez Nuño 22310284 6E2
"""
Created on Wed Feb 12 23:21:57 2025

@author: javal
"""

# Definir la lista de amigos
friends_info = {
    "Dario": 24,
    "Jacob": 21,
    "Eric": 33
}

# Función para imprimir la información 
def print_friends_info(friends_dict):
    print("\nInformación de amigos:")
    for name, age in friends_dict.items():
        print(f"{name} tiene {age} años.")

# Función para actualizar la edad de un amigo
def update_friend_age(friends_dict):
    while True:
        name = input("Ingrese el nombre del amigo cuya edad desea actualizar (o escriba 'salir' para terminar): ") #Solicitar el nombre del amigo al cual modificar la edad
        if name.lower() == 'salir': #establecer salir como opcion para que termine el programa
            print("Saliendo del programa...")
            break
        if name in friends_dict:
            new_age = int(input(f"Ingrese la nueva edad de {name}: ")) #soliciar nueva edadd
            friends_dict[name] = new_age
            print(f"Edad de {name} actualizada a {new_age} años.")
        else:
            print("Ese amigo no está en la lista.")
        print_friends_info(friends_dict)
        
#imprimir en panttalla
print_friends_info(friends_info)
update_friend_age(friends_info)
