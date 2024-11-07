import os
import numpy as np
from sklearn.cluster import KMeans
import json
from ClasificadorAudio import ClasificadorAudio
from ClasificadorImagen import ClasificadorImagen
from collections import Counter

class Entrenador:
    def __init__(self, k_vecinos=5, k_centroides=4, datos_procesados_path="saves/datos_procesados.json"):
        self.k_vecinos = k_vecinos
        self.k_centroides = k_centroides
        self.datos_procesados_path = datos_procesados_path

        # Clasificadores
        self.clasificador_audio = ClasificadorAudio()
        self.clasificador_imagen = ClasificadorImagen()

        # Datos de entrenamiento
        self.audios_entrenamiento = None
        self.imagenes_entrenamiento = None
        self.labels_audio_entrenamiento = None
        self.labels_imagen_entrenamiento = None

    def cargar_datos(self):
        """Carga los datos de entrenamiento desde el archivo de datos procesados."""
        if not os.path.exists(self.datos_procesados_path):
            print(f"Error: No se encontró el archivo de datos procesados en: {self.datos_procesados_path}")
            raise FileNotFoundError(f"Archivo de datos procesados '{self.datos_procesados_path}' no encontrado.")
        
        try:
            with open(self.datos_procesados_path, 'r') as f:
                datos = json.load(f)
                self.audios_entrenamiento = np.array(datos['audio'])
                self.labels_audio_entrenamiento = np.array(datos['etiquetas_audio'])
                self.imagenes_entrenamiento = np.array(datos['imagen'])
                self.labels_imagen_entrenamiento = np.array(datos['etiquetas_imagen'])
            print("Datos de entrenamiento cargados exitosamente desde el archivo.")
        except Exception as e:
            print(f"Error al cargar datos procesados: {e}")
            raise

    def entrenar_knn(self):
        """Entrena el clasificador K-NN para audios utilizando los datos cargados."""
        if self.audios_entrenamiento is None or self.audios_entrenamiento.size == 0:
            print("Error: No hay datos de audio cargados para entrenar K-NN.")
            raise ValueError("No hay datos de audio cargados para entrenar K-NN.")
        
        # Asegúrate de que las etiquetas de audio estén disponibles
        if self.labels_audio_entrenamiento is None or len(self.labels_audio_entrenamiento) != len(self.audios_entrenamiento):
            print("Error: Las etiquetas de audio no están disponibles o no coinciden con las características.")
            raise ValueError("Las etiquetas de audio no están disponibles o no coinciden con las características.")
        
        self.clasificador_audio.cargar_datos_entrenamiento(self.audios_entrenamiento, self.labels_audio_entrenamiento)
        print("Clasificador K-NN para audio configurado y entrenado.")

    def entrenar_kmeans(self):
        """Entrena el clasificador K-means para imágenes utilizando los datos cargados."""
        if self.imagenes_entrenamiento is None or self.imagenes_entrenamiento.size == 0:
            print("Error: No hay datos de imagen cargados para entrenar K-means.")
            raise ValueError("No hay datos de imagen cargados para entrenar K-means.")
        
        if self.labels_imagen_entrenamiento is None or len(self.labels_imagen_entrenamiento) != len(self.imagenes_entrenamiento):
            print("Error: Las etiquetas de imagen no están disponibles o no coinciden con las características.")
            raise ValueError("Las etiquetas de imagen no están disponibles o no coinciden con las características.")
        
        n_samples, n_features = self.imagenes_entrenamiento.shape
        max_iter = 300
        tol = 1e-4

        # Paso 1: Inicialización inspirada en K-means++
        #Selecciona un punto aleatorio de imagenes_entrenamiento como el primer centroide.
        #Para cada centroide adicional, calcula la distancia mínima de cada muestra a los centroides existentes
        # y selecciona un nuevo centroide basado en una distribución de probabilidad proporcional a estas distancias.
        centroides = [self.imagenes_entrenamiento[np.random.randint(0, n_samples)]]  # Selecciona un punto muestra aleatorio como primer centroide
        for _ in range(1, self.k_centroides):
            distancias = np.array([
                min(np.sum((muestra - centroide) ** 2) for centroide in centroides) # Calcula la distancia euclidiana mínima de cada muestra a los centroides existentes
                for muestra in self.imagenes_entrenamiento
            ]) # distancias -> array de distancias mínimas de cada muestra a los centroides existentes
            probabilidades = distancias / distancias.sum() # Normaliza las distancias para obtener probabilidades, donde cada probabildad es proporcional a la distancia mínima de una muestra a los centroides existentes
            nuevo_centroide = self.imagenes_entrenamiento[np.random.choice(n_samples, p=probabilidades)] # Selecciona un nuevo centroide basado en una distribución de probabilidad proporcional a las distancias
            centroides.append(nuevo_centroide)
        
        centroides = np.array(centroides)
        
        for _ in range(max_iter):
            # Paso 2: Asigna cada punto al centroide más cercano
            etiquetas = np.array([
                np.argmin([np.sum((muestra - centroide) ** 2) for centroide in centroides])
                for muestra in self.imagenes_entrenamiento
            ])

            # Paso 3: Calcula nuevos centroides como la media de las muestras asignadas a cada centroide!!!!!
            nuevos_centroides = np.array([
                self.imagenes_entrenamiento[etiquetas == i].mean(axis=0) if np.any(etiquetas == i) else centroides[i]
                for i in range(self.k_centroides)
            ])

            # Paso 4: Verificar la convergencia - Norma entre controides antiguos y nuevos
            suma_cuadrados = 0.0
            for i in range(self.k_centroides):
                diferencia = centroides[i] - nuevos_centroides[i]
                suma_cuadrados += np.sum(diferencia ** 2)
            desplazamiento = np.sqrt(suma_cuadrados)
            if desplazamiento < tol:
                break
            centroides = nuevos_centroides


        etiquetas_clusters = self.asignar_etiquetas_a_centroides(etiquetas)
        self.clasificador_imagen.cargar_centroides(centroides, etiquetas_clusters)
        print("Clasificador K-means para imágenes configurado y entrenado.")
    
    def asignar_etiquetas_a_centroides(self, labels_kmeans):
        """
        Asigna una etiqueta a cada centroide basada en la etiqueta más frecuente en su cluster.

        :param labels_kmeans: Array de etiquetas asignadas por K-means a cada muestra.
        :return: Lista de etiquetas asignadas a cada centroide.
        """
        etiquetas_centroides = []
        for i in range(self.k_centroides):
            indices = np.where(labels_kmeans == i)[0]
            etiquetas_cluster = self.labels_imagen_entrenamiento[indices]
            if len(etiquetas_cluster) == 0:
                etiqueta_mas_comun = "Sin_Etiqueta"
            else:
                etiqueta_mas_comun = Counter(etiquetas_cluster).most_common(1)[0][0]
            etiquetas_centroides.append(etiqueta_mas_comun)
            print(f"Centroide {i}: Etiqueta asignada '{etiqueta_mas_comun}'")
        return etiquetas_centroides

    def configurar_clasificadores(self):
        """Configura y entrena ambos clasificadores: K-NN para audios y K-means para imágenes."""
        self.entrenar_knn()
        self.entrenar_kmeans()
        print("Clasificadores configurados y entrenados.")

    def guardar_modelos(self, ruta_modelo):
        modelos = {
            'clasificador_audio': self.clasificador_audio.to_dict(),
            'clasificador_imagen': self.clasificador_imagen.to_dict()  # Guardar centroides y etiquetas
        }
        try:
            with open(ruta_modelo, 'w') as f:
                json.dump(modelos, f, indent=4)
            print(f"Modelos entrenados guardados exitosamente en {ruta_modelo}.")
        except Exception as e:
            print(f"Error al guardar los modelos en {ruta_modelo}: {e}")
            raise

    def cargar_modelos(self, ruta_modelo):
        if not os.path.exists(ruta_modelo):
            print(f"Error: No se encontró el archivo de modelo en la ruta especificada: {ruta_modelo}")
            raise FileNotFoundError(f"No se encontró el archivo de modelo en la ruta especificada: {ruta_modelo}")
        
        try:
            with open(ruta_modelo, 'r') as f:
                modelos = json.load(f)
                # Cargar clasificador de audio
                self.clasificador_audio.from_dict(modelos['clasificador_audio'])
                # Cargar clasificador de imagen
                self.clasificador_imagen.from_dict(modelos['clasificador_imagen'])
            print(f"Modelos entrenados cargados exitosamente desde {ruta_modelo}.")
        except json.JSONDecodeError:
            print(f"Error: El archivo de modelo {ruta_modelo} está corrupto o no es válido.")
            raise
        except Exception as e:
            print(f"Se produjo un error inesperado al cargar los modelos: {e}")
            raise