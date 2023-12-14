import sys
import pandas as pd
import requests

def descargar_y_guardar_csv(url, nombre_archivo):
    respuesta = requests.get(url)

    if respuesta.status_code == 200:
        with open(nombre_archivo, 'w') as archivo:
            archivo.write(respuesta.text)
        print(f"Descarga exitosa. Datos guardados en {nombre_archivo}")
    else:
        print(f"Error al descargar los datos. Código de respuesta: {respuesta.status_code}")

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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python automatizar_proceso.py <url>")
        sys.exit(1)

    url_datos = sys.argv[1]

    descargar_y_guardar_csv(url_datos, "datos_descargados.csv")

    df = pd.read_csv("datos_descargados.csv")

    limpieza_y_preparacion(df)

#python automatizar_proceso.py https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv
