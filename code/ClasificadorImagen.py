import numpy as np

class ClasificadorImagen:
    def __init__(self):
        self.centroides = None
        self.etiquetas_centroides = None

    def cargar_centroides(self, centroides, etiquetas_centroides=None):
        """Carga los centroides calculados por K-means y sus etiquetas correspondientes."""
        self.centroides = np.array([list(centroide) for centroide in centroides])
        self.etiquetas_centroides = etiquetas_centroides
        if etiquetas_centroides is not None:
            print("Centroides y sus etiquetas cargados exitosamente.")
        else:
            print("Centroides cargados sin etiquetas.")

    def predecir(self, caracteristicas_imagen):
        """
        Predice la etiqueta de la imagen basándose en el centroide más cercano.

        :param caracteristicas_imagen: Array de características extraídas de la imagen a clasificar.
        :return: Etiqueta del centroide más cercano.
        """
        if self.centroides is None:
            raise ValueError("Los centroides no han sido cargados. Usa 'cargar_centroides' primero.")
        
        # Calcular las distancias euclidianas manualmente
        distancias = []
        for centroide in self.centroides:
            distancia = 0
            for a, b in zip(centroide, caracteristicas_imagen):
                distancia += (a - b) ** 2
            distancia = distancia ** 0.5  # Raíz cuadrada de la suma de los cuadrados
            distancias.append(distancia)
        
        # Encontrar el índice del centroide más cercano
        indice_cercano = distancias.index(min(distancias))
        
        if self.etiquetas_centroides is not None:
            return self.etiquetas_centroides[indice_cercano]
        else:
            return indice_cercano  # Retorna el índice si no hay etiquetas

    def to_dict(self):
        """Convierte el clasificador a un diccionario serializable en JSON."""
        return {
            "centroides": self.centroides.tolist() if self.centroides is not None else None,
            "etiquetas_centroides": self.etiquetas_centroides if self.etiquetas_centroides is not None else None
        }

    def from_dict(self, data):
        """Carga los datos del clasificador desde un diccionario."""
        self.centroides = np.array(data["centroides"]) if data.get("centroides") is not None else None
        self.etiquetas_centroides = data.get("etiquetas_centroides", None)
        if self.centroides is not None:
            print("Centroides y etiquetas cargados desde el diccionario.")
        else:
            print("Centroides no encontrados en el diccionario.")