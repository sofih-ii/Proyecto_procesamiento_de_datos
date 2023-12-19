import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carga los datos desde el enlace
df = pd.read_csv('https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv')

# Elimina las columnas DEATH_EVENT, age y categoria_edad
X = df.drop(columns=['DEATH_EVENT', 'age', 'categoria_edad'], errors='ignore')

# Verifica si la columna 'age' está presente antes de intentar acceder a ella
if 'age' in df.columns:
    # Vector y (columna age)
    y = df['age'].values

    # Ajusta una regresión lineal sobre el resto de columnas
    model = LinearRegression()
    model.fit(X, y)

    # Predice las edades
    y_pred = model.predict(X)

    # Calcula el error cuadrático medio
    mse = mean_squared_error(y, y_pred)
    
    print(f"Error Cuadrático Medio: {mse}")
else:
    print("La columna 'age' no está presente en el DataFrame.")
