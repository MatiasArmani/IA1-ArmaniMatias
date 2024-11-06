import os
import json
from ProcesadorAudio import ProcesadorAudio
from ProcesadorImagen import ProcesadorImagen

class Procesador:
    def __init__(self, rutas_db):
        """
        Inicializa el procesador general.

        :param rutas_db: Lista de rutas a las carpetas que contienen archivos de audio e imagen.
        """
        self.rutas_db = rutas_db
        self.datos_audio = []
        self.datos_imagen = []
        self.etiquetas_audio = []
        self.etiquetas_imagen = []
        self.errores_audio = 0
        self.errores_imagen = 0
        self.audios_exitosos = 0
        self.imagenes_exitosas = 0
        self.archivos_audio_error = []
        self.archivos_imagen_error = []

    def obtener_archivos_audio(self, carpeta):
        """Obtiene una lista de archivos de audio en la carpeta especificada."""
        archivos_audio = [os.path.join(carpeta, archivo) for archivo in os.listdir(carpeta) if archivo.endswith('.wav')]
        print(f"Archivos de audio encontrados en {carpeta}: {len(archivos_audio)}")
        return archivos_audio

    def obtener_archivos_imagen(self, carpeta):
        """Obtiene una lista de archivos de imagen en la carpeta especificada."""
        archivos_imagen = [os.path.join(carpeta, archivo) for archivo in os.listdir(carpeta) if archivo.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Archivos de imagen encontrados en {carpeta}: {len(archivos_imagen)}")
        return archivos_imagen

    def obtener_etiqueta(self, carpeta):
        """Genera la etiqueta basada en el nombre de la carpeta."""
        return os.path.basename(os.path.normpath(carpeta))

    def procesar_audios(self, carpeta):
        """Procesa todos los archivos de audio en una carpeta específica."""
        etiqueta = self.obtener_etiqueta(carpeta)
        archivos_audio = self.obtener_archivos_audio(carpeta)
        if not archivos_audio:
            print(f"Advertencia: No se encontraron archivos de audio en {carpeta}.")
            return

        for archivo_audio in archivos_audio:
            print(f"Procesando audio: {archivo_audio}")
            try:
                procesador_audio = ProcesadorAudio(archivo_audio)
                procesador_audio.cargar_audio()
                procesador_audio.preprocesar_audio()
                procesador_audio.extraer_caracteristicas()
                
                self.datos_audio.append(procesador_audio.caracteristicas.tolist()) 
                self.etiquetas_audio.append(etiqueta)
                self.audios_exitosos += 1
                print(f"Características de audio extraídas: {procesador_audio.caracteristicas}")
            except Exception as e:
                print(f"Error al procesar el audio {archivo_audio}: {e}")
                self.errores_audio += 1
                self.archivos_audio_error.append(archivo_audio)

    def procesar_imagenes(self, carpeta):
        """Procesa todos los archivos de imagen en una carpeta específica."""
        etiqueta = self.obtener_etiqueta(carpeta)
        archivos_imagen = self.obtener_archivos_imagen(carpeta)
        if not archivos_imagen:
            print(f"Advertencia: No se encontraron archivos de imagen en {carpeta}.")
            return

        for archivo_imagen in archivos_imagen:
            print(f"Procesando imagen: {archivo_imagen}")
            try:
                procesador_imagen = ProcesadorImagen(archivo_imagen)
                procesador_imagen.cargar_imagen()
                procesador_imagen.aplicar_retoque_lab()
                procesador_imagen.eliminar_fondo()
                procesador_imagen.extraer_caracteristicas()
                
                self.datos_imagen.append(procesador_imagen.caracteristicas.tolist())
                self.etiquetas_imagen.append(etiqueta)
                self.imagenes_exitosas += 1
                print(f"Características de imagen extraídas: {procesador_imagen.caracteristicas}")
            except Exception as e:
                print(f"Error al procesar la imagen {archivo_imagen}: {e}")
                self.errores_imagen += 1
                self.archivos_imagen_error.append(archivo_imagen)

    def procesar_varias_carpetas(self):
        """Procesa archivos de audio e imagen en todas las carpetas especificadas."""
        for carpeta in self.rutas_db:
            print(f"\nProcesando carpeta: {carpeta}")
            self.procesar_audios(carpeta)
            self.procesar_imagenes(carpeta)

    def guardar_datos(self):
        """Guarda los datos de audio e imagen procesados en un archivo JSON."""
        datos = {
            "audio": self.datos_audio,
            "imagen": self.datos_imagen,
            "etiquetas_audio": self.etiquetas_audio,
            "etiquetas_imagen": self.etiquetas_imagen
        }
        with open("datos_procesados.json", "w") as f:
            json.dump(datos, f, indent=4)
        print("Datos procesados guardados en: datos_procesados.json")

    def mostrar_resumen(self):
        """Muestra un resumen del procesamiento de los archivos."""
        print("\nResumen de procesamiento:")
        print(f"Audios procesados exitosamente: {self.audios_exitosos}")
        print(f"Audios con errores: {self.errores_audio}")
        if self.archivos_audio_error:
            print("Archivos de audio con errores:")
            for archivo in self.archivos_audio_error:
                print(f" - {archivo}")
                
        print(f"Imágenes procesadas exitosamente: {self.imagenes_exitosas}")
        print(f"Imágenes con errores: {self.errores_imagen}")
        if self.archivos_imagen_error:
            print("Archivos de imagen con errores:")
            for archivo in self.archivos_imagen_error:
                print(f" - {archivo}")

if __name__ == "__main__":
    ruta_real = input("Ingrese la ruta de la carpeta para probar: ").strip()

    print("Iniciando procesamiento")
    procesador = Procesador([ruta_real])
    procesador.procesar_varias_carpetas()
    procesador.mostrar_resumen()
    procesador.guardar_datos()
    print("Procesamiento completo")