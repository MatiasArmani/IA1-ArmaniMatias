from flask import Flask, request, jsonify, render_template, send_file, url_for
import os
import json
import tempfile
import threading
from Entrenador import Entrenador
from ProcesadorAudio import ProcesadorAudio
from ProcesadorImagen import ProcesadorImagen
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = "saves/modelos_entrenados.json"
TEMP_DIR = tempfile.gettempdir()
EXTENSIONES_AUDIO_PERMITIDAS = {'wav'}
EXTENSIONES_IMAGEN_PERMITIDAS = {'jpg', 'jpeg', 'png'}

# Variable global para el proceso del servidor
server_process = None

# Diccionario para almacenar las imágenes recibidas temporalmente con sus etiquetas predecidas
imagenes_temporales = {}

entrenador = Entrenador(datos_procesados_path="saves/datos_procesados.json")

def archivo_permitido(nombre_archivo, extensiones_permitidas):
    return '.' in nombre_archivo and nombre_archivo.rsplit('.', 1)[1].lower() in extensiones_permitidas

def cargar_modelos():
    """Carga los modelos entrenados desde el archivo JSON y actualiza los clasificadores en el entrenador."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el archivo de modelo en la ruta especificada: {MODEL_PATH}")
    try:
        with open(MODEL_PATH, 'r') as f:
            modelos = json.load(f)
            # Cargar clasificador de audio
            entrenador.clasificador_audio.from_dict(modelos['clasificador_audio'])
            # Cargar clasificador de imagen
            entrenador.clasificador_imagen.from_dict(modelos['clasificador_imagen'])
        print("Modelos cargados exitosamente desde JSON.")
    except json.JSONDecodeError:
        print(f"Error: El archivo de modelo {MODEL_PATH} está corrupto o no es válido.")
        raise
    except KeyError as e:
        print(f"Error: Falta la clave {e} en el archivo de modelo.")
        raise
    except Exception as e:
        print(f"Se produjo un error inesperado al cargar los modelos: {e}")
        raise

def iniciar_visualizacion_imagen(procesador_imagen):
    """Función para iniciar la visualización en un hilo separado."""
    hilo_visualizacion = threading.Thread(target=procesador_imagen.visualizar_resultados)
    hilo_visualizacion.start()

def iniciar_interfaz(procesador_audio):
    """Inicia interfaz Tkinter para reproducir el audio procesado en hilo separado."""

    hilo_interfaz = threading.Thread(target=procesador_audio.mostrar_interfaz)
    hilo_interfaz.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clasificar_audio', methods=['POST'])
def clasificar_audio():
    archivo_audio = request.files.get('audio')
    if archivo_audio and archivo_permitido(archivo_audio.filename, EXTENSIONES_AUDIO_PERMITIDAS):
        try:
            cargar_modelos()
        except Exception as e:
            return jsonify({'error': f"Error al cargar los modelos: {str(e)}"}), 500

        filename = secure_filename(archivo_audio.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=filename, dir=TEMP_DIR) as temp_audio:
            archivo_audio.save(temp_audio.name)
            temp_audio_path = temp_audio.name

        try:
            procesador_audio = ProcesadorAudio(temp_audio_path)
            procesador_audio.cargar_audio()
            procesador_audio.preprocesar_audio()
            procesador_audio.extraer_caracteristicas()
            iniciar_interfaz(procesador_audio)

            # Realizar la predicción
            prediccion = entrenador.clasificador_audio.predecir(procesador_audio.caracteristicas)
            etiqueta_predicha = str(prediccion)

            # Generar URLs para las imágenes que coinciden con la etiqueta predicha
            imagenes_mismas_etiquetas = [
                url_for('mostrar_imagen', path=img_path)
                for label, img_path in imagenes_temporales.items() if label == etiqueta_predicha
            ]

            # Renderizar la plantilla HTML con la etiqueta y las imágenes coincidentes
            return render_template('resultado_audio.html', prediccion=etiqueta_predicha, imagenes=imagenes_mismas_etiquetas)

        except Exception as e:
            return jsonify({'error': f"Error al procesar el audio: {str(e)}"}), 500
        finally:
            os.remove(temp_audio_path)
    else:
        return jsonify({'error': 'No se proporcionó un archivo de audio válido.'}), 400

@app.route('/clasificar_imagen', methods=['POST'])
def clasificar_imagen():
    archivo_imagen = request.files.get('imagen')
    if archivo_imagen and archivo_permitido(archivo_imagen.filename, EXTENSIONES_IMAGEN_PERMITIDAS):
        try:
            cargar_modelos()
        except Exception as e:
            return jsonify({'error': f"Error al cargar los modelos: {str(e)}"}), 500

        filename = secure_filename(archivo_imagen.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=filename, dir=TEMP_DIR) as temp_image:
            archivo_imagen.save(temp_image.name)
            temp_image_path = temp_image.name

        try:
            procesador_imagen = ProcesadorImagen(temp_image_path)
            procesador_imagen.cargar_imagen()
            procesador_imagen.aplicar_retoque_lab()
            procesador_imagen.eliminar_fondo()
            procesador_imagen.extraer_caracteristicas()

            iniciar_visualizacion_imagen(procesador_imagen)

            prediccion = entrenador.clasificador_imagen.predecir(procesador_imagen.caracteristicas)
            etiqueta_predicha = str(prediccion)

            # Almacenar la imagen con su etiqueta y renderizar la página de resultado
            imagenes_temporales[etiqueta_predicha] = temp_image_path
            return render_template('resultado_imagen.html', prediccion=etiqueta_predicha)
            
        except Exception as e:
            return jsonify({'error': f"Error al procesar la imagen: {str(e)}"}), 500
    else:
        return jsonify({'error': 'No se proporcionó un archivo de imagen válido.'}), 400
    
@app.route('/mostrar_imagen')
def mostrar_imagen():
    path = request.args.get('path')
    if path and os.path.exists(path):
        return send_file(path, mimetype='image/jpeg')
    else:
        return "Imagen no encontrada", 404


if __name__ == '__main__':
    try:
        cargar_modelos()
    except Exception as e:
        print(f"Error al cargar los modelos al iniciar el servidor: {e}")
        exit(1)
    
    app.run(host='0.0.0.0', port=5000, threaded=True)