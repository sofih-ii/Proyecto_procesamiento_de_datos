import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv')

# Elimina la columna 'categoria_edad' si existe
df = df.drop(columns=['categoria_edad'], errors='ignore')

# Define X (atributos) y y (etiquetas)
X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']

# Realiza la partición estratificada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Crea el modelo
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Ajusta el modelo
rf_model.fit(X_train, y_train)

# Predice en el conjunto de prueba
y_pred = rf_model.predict(X_test)

# Calcula la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Calcula la exactitud (accuracy)
accuracy = accuracy_score(y_test, y_pred)

# Calcula el F1-Score
f1 = f1_score(y_test, y_pred)

# Imprime los resultados
print("Matriz de Confusión:")
print(conf_matrix)
print("Accuracy:", accuracy)
print("F1-Score:", f1)
