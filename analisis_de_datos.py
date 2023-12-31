import pandas as pd
from datasets import load_dataset

dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

df = pd.DataFrame(data)

personas_fallecidas = df[df["is_dead"] == 1]
personas_no_fallecidas = df[df["is_dead"] == 0]

promedio_edad_fallecidos = personas_fallecidas["age"].mean()
promedio_edad_no_fallecidos = personas_no_fallecidas["age"].mean()

print(f"Promedio de Edad Fallecidos: {promedio_edad_fallecidos}")
print(f"Promedio de Edad No Fallecidos: {promedio_edad_no_fallecidos}")

tipos_de_datos = df.dtypes
print("\nTipos de Datos:")
print(tipos_de_datos)

# Verificar la presencia de 'is_male' e 'is_smoker' antes de agrupar
if 'is_male' in df.columns and 'is_smoker' in df.columns:
    fumadores_por_genero = df.groupby(['is_male', 'is_smoker']).size().unstack()
    print("\nCantidad de Hombres y Mujeres Fumadores:")
    print(fumadores_por_genero)
else:
    print("Las columnas 'is_male' y/o 'is_smoker' no están presentes en el DataFrame.")
