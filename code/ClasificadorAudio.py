import numpy as np

class ClasificadorAudio:
    def __init__(self, k=5):
        self.k = k
        self.audios_entrenamiento = None
        self.labels_audio_entrenamiento = None

    def cargar_datos_entrenamiento(self, audios, labels):
        """Carga los datos de entrenamiento para K-NN"""
        self.audios_entrenamiento = np.array([list(audio) for audio in audios])
        self.labels_audio_entrenamiento = np.array(list(labels))
        print("Datos de entrenamiento cargados para el modelo K-NN.")

    def predecir(self, caracteristicas_audio):
        """Predice la etiqueta de un nuevo audio basado en el modelo K-NN entrenado."""
        if self.audios_entrenamiento is None or self.labels_audio_entrenamiento is None:
            raise ValueError("El modelo K-NN no ha sido entrenado. Usa 'cargar_datos_entrenamiento' primero.")

        # Calcular las distancias euclidianas manualmente con comprensión de listas
        distancias = [
            sum((a - b) ** 2 for a, b in zip(audio_entrenamiento, caracteristicas_audio)) ** 0.5
            for audio_entrenamiento in self.audios_entrenamiento
        ]

        # Obtener los índices de los k vecinos más cercanos
        indices_vecinos = sorted(range(len(distancias)), key=lambda i: distancias[i])[:self.k]

        # Obtener las etiquetas de los k vecinos más cercanos
        etiquetas_vecinos = [self.labels_audio_entrenamiento[i] for i in indices_vecinos]

        # Contar la frecuencia de cada etiqueta
        conteo_etiquetas = {}
        for etiqueta in etiquetas_vecinos:
            if etiqueta in conteo_etiquetas:
                conteo_etiquetas[etiqueta] += 1
            else:
                conteo_etiquetas[etiqueta] = 1

        # Devolver la etiqueta más común
        etiqueta_predicha = max(conteo_etiquetas, key=conteo_etiquetas.get)
        return etiqueta_predicha

    def to_dict(self):
        """Convierte el clasificador a un diccionario serializable en JSON."""
        return {
            "k": self.k,
            "audios_entrenamiento": self.audios_entrenamiento.tolist() if self.audios_entrenamiento is not None else None,
            "labels_audio_entrenamiento": self.labels_audio_entrenamiento.tolist() if self.labels_audio_entrenamiento is not None else None
        }

    def from_dict(self, data):
        """Carga los datos del clasificador desde un diccionario"""
        self.k = data.get("k", 5)
        self.audios_entrenamiento = np.array(data["audios_entrenamiento"]) if data.get("audios_entrenamiento") is not None else None
        self.labels_audio_entrenamiento = np.array(data["labels_audio_entrenamiento"]) if data.get("labels_audio_entrenamiento") is not None else None
        if self.audios_entrenamiento is not None and self.labels_audio_entrenamiento is not None:
            print("Datos de entrenamiento cargados desde el diccionario para el modelo K-NN.")
        else:
            print("Advertencia: No se pudieron cargar los datos de entrenamiento para el clasificador de audio.")