import pandas as pd

def limpieza_y_preparacion(datos):
    faltantes = datos.isnull().sum()
    print("Valores Faltantes:")
    print(faltantes)

    datos = datos.drop_duplicates()

    datos = datos[datos['age'] >= 0]

    bins = [0, 12, 19, 39, 59, float('inf')]
    labels = ['Niño', 'Adolescente', 'Joven Adulto', 'Adulto', 'Adulto Mayor']
    datos['edad_categoria'] = pd.cut(datos['age'], bins=bins, labels=labels, right=False)

    datos.to_csv("datos_limpio_y_preparado.csv", index=False)

    print("Limpieza y preparación completada. Resultados guardados en datos_limpio_y_preparado.csv")

df = pd.read_csv("heart_failure_dataset.csv")
limpieza_y_preparacion(df)
