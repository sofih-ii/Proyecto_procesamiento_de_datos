import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datos_limpio_y_preparado.csv")

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Distribuciones', fontsize=16)

anemico_counts = df['anaemia'].value_counts()
axs[0, 0].pie(anemico_counts, labels=anemico_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
axs[0, 0].set_title('Distribución de Anémicos')

diabetico_counts = df['diabetes'].value_counts()
axs[0, 1].pie(diabetico_counts, labels=diabetico_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
axs[0, 1].set_title('Distribución de Diabéticos')

fumador_counts = df['smoking'].value_counts()
axs[1, 0].pie(fumador_counts, labels=fumador_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
axs[1, 0].set_title('Distribución de Fumadores')

muerto_counts = df['DEATH_EVENT'].value_counts()
axs[1, 1].pie(muerto_counts, labels=muerto_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
axs[1, 1].set_title('Distribución de Muertos')

plt.show()
