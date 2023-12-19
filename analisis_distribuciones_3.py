import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px

# Carga los datos
df = pd.read_csv('https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv')

# Exporta una matriz con sólo los valores de los atributos en formato de numpy array
X = df.drop(columns=['DEATH_EVENT']).values

# Exporta un array unidimensional de sólo la columna objetivo DEATH_EVENT
y = df['DEATH_EVENT'].values

# Realiza la reducción de dimensionalidad con t-SNE
X_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(X)

# Convierte los datos reducidos a un DataFrame para facilitar el trazado
df_embedded = pd.DataFrame(X_embedded, columns=['Componente 1', 'Componente 2', 'Componente 3'])
df_embedded['DEATH_EVENT'] = y

# Realiza un gráfico de dispersión 3D con Plotly
fig = px.scatter_3d(df_embedded, x='Componente 1', y='Componente 2', z='Componente 3', color='DEATH_EVENT')
fig.show()
