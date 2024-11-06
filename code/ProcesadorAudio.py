import numpy as np
import os
import pyaudio
import tkinter as tk
from scipy.io import wavfile
import scipy.signal as signal
from python_speech_features import mfcc
import librosa
import librosa.display
import matplotlib.pyplot as plt
import threading

class ProcesadorAudio:
    def __init__(self, ruta_audio):
        """param: ruta_audio: Ruta al archivo de audio (.wav)."""
        self.ruta_audio = ruta_audio
        self.audio = None
        self.tasa_muestreo = None
        self.caracteristicas = None
        self.audio_recortado = None
        self.audio_final = None

        # Versiones normalizadas para reproducción
        self.audio_original_normalizado = None
        self.audio_recortado_normalizado = None

        # Características adicionales
        self.caracteristicas_mfcc = None
        self.caracteristicas_spectral_contrast = None

    def cargar_audio(self):
        """Carga el audio desde la ruta especificada usando scipy.io.wavfile."""
        try:
            self.tasa_muestreo, audio = wavfile.read(self.ruta_audio)

            # Verificar si el audio es estéreo y seleccionar un solo canal
            if audio.ndim == 2:
                print("Audio estéreo detectado. Seleccionando el canal izquierdo para procesamiento.")
                self.audio = audio[:, 0]  # Selecciona el primer canal (izquierdo)
            else:
                self.audio = audio

            # Verificación audio en silencio
            if np.max(np.abs(self.audio)) == 0:
                print("Advertencia: El audio cargado está en silencio o tiene amplitud cero.")
            else:
                print("Audio cargado con contenido válido.")

            # Loguear estadísticas básicas del audio
            duracion = len(self.audio) / self.tasa_muestreo
            print(f"Duración del audio: {duracion:.2f} segundos.")

            # Crear versión normalizada
            self.audio_original_normalizado = self.normalizar_audio_para_reproduccion(self.audio)
            print(f"Original Normalizado: min={self.audio_original_normalizado.min()}, "
                  f"max={self.audio_original_normalizado.max()}, "
                  f"dtype={self.audio_original_normalizado.dtype}, "
                  f"length={len(self.audio_original_normalizado)}")

        except FileNotFoundError:
            print(f"Error: El archivo {self.ruta_audio} no fue encontrado.")
            raise
        except ValueError:
            print(f"Error: No se pudo leer el archivo {self.ruta_audio}. Asegúrate de que es un archivo WAV válido.")
            raise
        except Exception as e:
            print(f"Se produjo un error inesperado al cargar el audio: {e}")
            raise

    def filtrar_pasabajo(self, datos, frecuencia_corte=3000):
        frecuencia_nyquist = self.tasa_muestreo / 2.0
        corte_normalizado = frecuencia_corte / frecuencia_nyquist
        b, a = signal.butter(4, corte_normalizado, btype='low', analog=False)
        datos_filtrados = signal.lfilter(b, a, datos)
        print(f"Filtro pasa-bajo aplicado con frecuencia de corte: {frecuencia_corte} Hz.")
        return datos_filtrados

    def filtrar_pasaalto(self, datos, frecuencia_corte=300):
        frecuencia_nyquist = self.tasa_muestreo / 2.0
        corte_normalizado = frecuencia_corte / frecuencia_nyquist
        b, a = signal.butter(4, corte_normalizado, btype='high', analog=False)
        datos_filtrados = signal.lfilter(b, a, datos)
        print(f"Filtro pasa-alto aplicado con frecuencia de corte: {frecuencia_corte} Hz.")
        return datos_filtrados

    def normalizar_audio_para_reproduccion(self, datos):
        """Normaliza el audio a int16 - Cambio de formato."""
        if datos.dtype == np.int16:
            # Si ya está en int16, devolver sin cambios
            return datos
        elif np.issubdtype(datos.dtype, np.floating):
            # Si es float, normalizar a int16
            max_amp = np.max(np.abs(datos))
            if max_amp == 0:
                print("Advertencia: El audio tiene amplitud cero.")
                return datos.astype(np.int16)
            datos_normalizados = datos / max_amp * 32767 * 0.9  # Reducir ligeramente la ganancia para evitar clipping (Distorsión de la forma de onda)
            datos_normalizados = np.clip(datos_normalizados, -32768, 32767)
            return datos_normalizados.astype(np.int16)
        else:
            # Otros formatos, intentar convertir a int16
            datos_normalizados = datos.astype(np.int16)
            return datos_normalizados

    def normalizar_audio(self, datos):
        """Normaliza el audio para que su amplitud máxima sea 32767 (16-bit PCM)."""
        max_amp = np.max(np.abs(datos))
        if max_amp == 0:
            print("Advertencia: El audio tiene amplitud cero después de procesamiento.")
            return datos  # Devuelve el audio sin cambios si está en silencio
        datos_normalizados = datos / max_amp * 32767
        datos_normalizados = np.clip(datos_normalizados, -32768, 32767)  # Asegurar que esté en rango int16
        print(f"Audio normalizado: amplitud mínima {np.min(datos_normalizados)}, máxima {np.max(datos_normalizados)}")
        return datos_normalizados.astype(np.int16)  # Normalización a 16-bit PCM

    def preprocesar_audio(self):
        """Aplica recortes iniciales, filtros pasa-alto y pasa-bajo, y normaliza el audio."""
        if self.audio is None:
            raise ValueError("El audio no ha sido cargado. Llama a 'cargar_audio()' primero.")

        duracion_recorte = 0.25  # Duración en segundos para recortar
        muestras_recorte = int(duracion_recorte * self.tasa_muestreo)

        # Recortar el inicio y el final
        inicio = muestras_recorte
        fin = -muestras_recorte if muestras_recorte != 0 else None
        self.audio_recortado = self.audio[inicio:fin]

        # Verificar si el recorte no elimina todo el audio
        if len(self.audio_recortado) == 0:
            print("Advertencia: El recorte eliminó todo el audio.")
            self.audio_recortado = self.audio  # No recortar si el audio es demasiado corto
        else:
            print(f"Audio recortado: {len(self.audio_recortado)} muestras.")

        # Crear versión normalizada del audio recortado para reproducción
        self.audio_recortado_normalizado = self.normalizar_audio_para_reproduccion(self.audio_recortado)
        print(f"Recortado Normalizado: min={self.audio_recortado_normalizado.min()}, "
              f"max={self.audio_recortado_normalizado.max()}, "
              f"dtype={self.audio_recortado_normalizado.dtype}, "
              f"length={len(self.audio_recortado_normalizado)}")

        # Aplicar filtros
        audio_filtrado = self.filtrar_pasaalto(self.audio_recortado, frecuencia_corte=300)
        audio_filtrado = self.filtrar_pasabajo(audio_filtrado, frecuencia_corte=3000)

        # Normalizar
        self.audio_final = self.normalizar_audio(audio_filtrado)

        print("Preprocesamiento de audio completado.")

    def extraer_caracteristicas(self):
        """
        Extrae múltiples características del audio preprocesado:
        - MFCC
        - Spectral Contrast
        """
        if self.audio_final is None:
            raise ValueError("El audio no ha sido preprocesado. Llama a 'preprocesar_audio()' primero.")

        # Convertir audio a float para librosa
        audio_float = self.audio_final.astype(float)
        audio_float /= np.max(np.abs(audio_float))  # Normalizar entre -1 y 1

        # Extraer MFCC
        self.caracteristicas_mfcc = mfcc(self.audio_final, samplerate=self.tasa_muestreo, numcep=13, nfft=2048)
        self.caracteristicas_mfcc = np.mean(self.caracteristicas_mfcc, axis=0)
        print(f"MFCC extraídos: {self.caracteristicas_mfcc.shape}")

        # Extraer Spectral Contrast
        S = librosa.feature.melspectrogram(y=audio_float, sr=self.tasa_muestreo, n_fft=2048, hop_length=512)
        spectral_contrast = librosa.feature.spectral_contrast(S=S, sr=self.tasa_muestreo)
        self.caracteristicas_spectral_contrast = np.mean(spectral_contrast, axis=1)
        print(f"Spectral Contrast extraídas: {self.caracteristicas_spectral_contrast.shape}")

        # Mantengo solo las características que diferencian bien las clases: MFCC5, MFCC6, MFCC9, MFCC10 (indices 4,5,8,9) y Spectral Contrast2, Spectral Contrast5, Spectral Contrast6 (indices 1,4,5)  14, 17, 18

        # Concatena MFCC y Spectral Contrast
        self.caracteristicas = np.concatenate([
            self.caracteristicas_mfcc,            # 13
            self.caracteristicas_spectral_contrast  # 7
        ])
        print(f"Características combinadas antes de filtrar: {self.caracteristicas.shape}")  # (20)

        # Selecciona solo las características relevantes
        indices_a_mantener = [4, 5, 8, 9, 14, 17, 18]
        self.caracteristicas = self.caracteristicas[indices_a_mantener]
        print(f"Características combinadas filtradas: {self.caracteristicas.shape}")  # (7)

    def reproducir_audio(self, datos):
        try:
            p = pyaudio.PyAudio()

            formato = pyaudio.paInt16
            canales = 1  # Mono
            tasa_muestreo = self.tasa_muestreo
            stream = p.open(format=formato,
                            channels=canales,
                            rate=tasa_muestreo,
                            output=True)

            # Reproducir el audio
            datos_reproducir = datos.tobytes()
            stream.write(datos_reproducir)

            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Reproducción de audio finalizada.")
        except Exception as e:
            print(f"Error al reproducir audio: {e}")

    def reproducir_original(self):
        """Reproduce el audio original normalizado."""
        if self.audio_original_normalizado is not None:
            threading.Thread(target=self.reproducir_audio, args=(self.audio_original_normalizado,)).start()
        else:
            print("Audio original normalizado no disponible.")

    def reproducir_preprocesado(self):
        """Reproduce el audio filtrado y normalizado."""
        if self.audio_final is not None:
            threading.Thread(target=self.reproducir_audio, args=(self.audio_final,)).start()
        else:
            print("Audio preprocesado no disponible.")

    def mostrar_interfaz(self):
        ventana = tk.Tk()
        ventana.title("Reproductor de Audio Procesado")
        ventana.geometry("300x150")

        tk.Button(ventana, text="Reproducir Original", command=self.reproducir_original).pack(pady=10)
        tk.Button(ventana, text="Reproducir Filtrado y Normalizado", command=self.reproducir_preprocesado).pack(pady=10)
        ventana.mainloop()

if __name__ == "__main__":
    ruta_prueba = input("Ingrese la ruta del archivo para analizar: ").strip()

    if not os.path.exists(ruta_prueba):
        print(f"Archivo de prueba no encontrado en: {ruta_prueba}")
        exit(1)

    procesador = ProcesadorAudio(ruta_prueba)

    try:
        procesador.cargar_audio()
    except Exception as e:
        print(f"Error al cargar el audio: {e}")
        exit(1)

    try:
        procesador.preprocesar_audio()
    except Exception as e:
        print(f"Error al preprocesar el audio: {e}")
        exit(1)

    try:
        procesador.extraer_caracteristicas()
    except Exception as e:
        print(f"Error al extraer características: {e}")
        exit(1)

    procesador.mostrar_interfaz()