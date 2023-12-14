import requests

def descargar_y_guardar_csv(url, nombre_archivo):
    respuesta = requests.get(url)

    if respuesta.status_code == 200:
        with open(nombre_archivo, 'w') as archivo:
            archivo.write(respuesta.text)
        print(f"Descarga exitosa. Datos guardados en {nombre_archivo}")
    else:
        print(f"Error al descargar los datos. Código de respuesta: {respuesta.status_code}")

url_datos = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"
nombre_archivo_csv = "heart_failure_dataset.csv"
descargar_y_guardar_csv(url_datos, nombre_archivo_csv)
