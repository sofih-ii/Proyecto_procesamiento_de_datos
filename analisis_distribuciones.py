import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datos_limpio_y_preparado.csv")

plt.figure(figsize=(10, 6))
plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribución de Edades')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

grupos = ['sex', 'anaemia', 'diabetes', 'smoking', 'DEATH_EVENT']
nombres = ['Hombres', 'Mujeres']

for grupo in grupos:
    plt.figure(figsize=(10, 6))
    for i, valor in enumerate(df[grupo].unique()):
        subconjunto = df[df[grupo] == valor]
        plt.bar(subconjunto['sex'].unique() + i * 0.4, subconjunto['sex'].value_counts(), width=0.4, label=f'{valor}')
    
    plt.title(f'Histograma de {grupo} por Género')
    plt.xlabel('Género')
    plt.ylabel('Cantidad')
    plt.xticks([0.2, 1.2], nombres)
    plt.legend()
    plt.show()
