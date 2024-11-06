import questionary
from Procesador import Procesador
from Entrenador import Entrenador
from Evaluador import Evaluador
import subprocess
import json
import time
import os

# Rutas para guardar los archivos procesados y entrenados
PROCESSED_DATA_PATH = "saves/datos_procesados.json"
TRAINED_MODEL_PATH = "saves/modelos_entrenados.json"
EVALUATION_RESULTS_PATH = "saves/evaluacion_procesados.json"

# Variable global para el proceso del servidor
server_process = None

def procesar_datos():
    rutas_db = ["../db/papa/", "../db/zanahoria/", "../db/camote/", "../db/berenjena/"]
    procesador = Procesador(rutas_db)
    procesador.procesar_varias_carpetas()
    procesador.mostrar_resumen()
    procesador.guardar_datos()

def entrenar_modelos(numero_iteraciones=10):
    entrenador = Entrenador(k_vecinos=5, k_centroides=4, datos_procesados_path=PROCESSED_DATA_PATH)

    # Cargar datos procesados
    try:
        entrenador.cargar_datos()
        print("Prueba de carga de datos exitosa.")
    except Exception as e:
        print(f"Error en la prueba de carga de datos: {e}")
        return

    mejor_calidad = float('-inf')
    mejor_modelo_path = None

    for iteracion in range(1, numero_iteraciones + 1):
        print(f"\nIteración {iteracion} de entrenamiento y evaluación.")

        # Entrenar clasificadores
        try:
            entrenador.configurar_clasificadores()
            print("Configuración y entrenamiento de clasificadores exitosa.")
        except Exception as e:
            print(f"Error en la configuración de clasificadores: {e}")
            continue

        # Guardar modelos entrenados temporalmente
        modelo_temp_path = f"{TRAINED_MODEL_PATH}_temp_iter{iteracion}"
        try:
            entrenador.guardar_modelos(modelo_temp_path)
            print(f"Modelo temporal guardado en: {modelo_temp_path}")
        except Exception as e:
            print(f"Error al guardar modelos en la iteración {iteracion}: {e}")
            continue

        # Evaluar calidad del modelo
        try:
            evaluador = Evaluador(modelo_path=modelo_temp_path, datos_procesados_path=PROCESSED_DATA_PATH)
            calidad = evaluador.evaluar_imagen()
            print(f"Calidad del modelo temporal {iteracion}: {calidad}")
        except Exception as e:
            print(f"Error al evaluar el modelo en la iteración {iteracion}: {e}")
            continue

        # Comparar con la mejor calidad obtenida hasta ahora
        if calidad > mejor_calidad:
            mejor_calidad = calidad
            mejor_modelo_path = modelo_temp_path
            print("Mejora en la calidad del modelo encontrada.")
        else:
            print("No hubo mejora en la calidad del modelo.")

    # Guardar el mejor modelo encontrado como TRAINED_MODEL_PATH
    if mejor_modelo_path is not None:
        try:
            import shutil
            shutil.copyfile(mejor_modelo_path, TRAINED_MODEL_PATH)
            print(f"\nMejor modelo guardado en: {TRAINED_MODEL_PATH}")
            print(f"Calidad del mejor modelo: {mejor_calidad}")
        except Exception as e:
            print(f"Error al guardar el mejor modelo: {e}")
    else:
        print("No se encontró un modelo con mejor calidad.")

    # Limpieza de modelos temporales
    try:
        import os
        for iteracion in range(1, numero_iteraciones + 1):
            modelo_temp_path = f"{TRAINED_MODEL_PATH}_temp_iter{iteracion}"
            if os.path.exists(modelo_temp_path):
                os.remove(modelo_temp_path)
                print(f"Modelo temporal eliminado: {modelo_temp_path}")
    except Exception as e:
        print(f"Error al eliminar modelos temporales: {e}")

def evaluar_modelos():
    # Verificar que el archivo de modelos exista
    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"Error: El archivo de modelos entrenados '{TRAINED_MODEL_PATH}' no existe. Por favor, entrena los modelos primero.")
        return

    evaluador = Evaluador(modelo_path=TRAINED_MODEL_PATH, datos_procesados_path=PROCESSED_DATA_PATH)

    print("Evaluación completa de los clasificadores:")
    evaluador.ejecutar_evaluacion()

    print("Visualización de características de audio:")
    evaluador.plot_3d_caracteristicas(tipo='audio')

    print("Visualización de características de imagen:")
    evaluador.plot_3d_caracteristicas(tipo='imagen')

    print("Cálculo de estadísticas por etiqueta para audio:")
    evaluador.calcular_estadisticas_por_etiqueta(evaluador.caracteristicas_audio, evaluador.labels_audio, tipo='audio')

    print("Cálculo de estadísticas por etiqueta para imagen:")
    evaluador.calcular_estadisticas_por_etiqueta(evaluador.caracteristicas_imagen, evaluador.labels_imagen, tipo='imagen')

def iniciar_servidor():
    global server_process
    if server_process is None:
        try:
            # Inicia el servidor en un proceso separado
            server_process = subprocess.Popen(["python", "servidor.py"])
            time.sleep(1)
            print("Servidor iniciado en http://localhost:5000")
        except Exception as e:
            print(f"Error al iniciar el servidor: {e}")
            server_process = None
    else:
        print("El servidor ya está en ejecución.")

def detener_servidor():
    global server_process
    if server_process is not None:
        server_process.terminate()
        server_process = None
        print("Servidor detenido.")
    else:
        print("El servidor no está en ejecución.")

def main():
    while True:
        opcion = questionary.select(
            "Seleccione una opción:",
            choices=[
                "Procesar datos",
                "Entrenar modelos",
                "Evaluar modelos",
                "Iniciar servidor",
                "Salir"
            ]
        ).ask()

        if opcion == "Procesar datos":
            procesar_datos()
        elif opcion == "Entrenar modelos":
            entrenar_modelos()
        elif opcion == "Evaluar modelos":
            evaluar_modelos()
        elif opcion == "Iniciar servidor":
            iniciar_servidor()
        elif opcion == "Salir":
            print("Saliendo del programa.")
            detener_servidor()
            break

if __name__ == "__main__":
    main()