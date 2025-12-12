import pandas as pd

# Leer el CSV
try:
    df = pd.read_csv("/Users/admin/Desktop/Github/Quadcode/Quadcodex_Team_Back/Data/usuarios_ejercicios_valoraciones.csv")
except FileNotFoundError:
    print(f"Archivo no encontrado: {"usuarios_ejercicios_valoraciones.csv"}")
    exit()

# Verificar que la columna 'valoracion' exista
if 'id_usuario' not in df.columns:
    print("La columna 'id_usuario' no existe en el archivo.")
    exit()

# Encontrar el valor más frecuente
valor_mas_frecuente = df['id_usuario'].mode()[0]
repeticiones = df['id_usuario'].value_counts().iloc[0]

print(f"El valor que más se repite es: {valor_mas_frecuente} (se repite {repeticiones} veces)")
