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
