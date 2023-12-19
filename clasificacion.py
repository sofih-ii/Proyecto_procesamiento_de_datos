import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el DataFrame desde el archivo CSV
df = pd.read_csv('https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv')

# Eliminar la columna 'categoria_edad' si existe
df = df.drop(columns=['categoria_edad'], errors='ignore')

# Visualizar la distribución de clases
plt.figure(figsize=(6, 4))
sns.countplot(x='DEATH_EVENT', data=df)
plt.title('Distribución de Clases')
plt.show()

# Eliminar la columna 'DEATH_EVENT' y 'categoria_edad' si existe
X = df.drop(columns=['DEATH_EVENT']).values
y = df['DEATH_EVENT'].values

# Realizar la partición estratificada del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Ajustar un árbol de decisión
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de test
y_pred = tree_model.predict(X_test)

# Calcular la accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy del Árbol de Decisión: {accuracy}')
