import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from sklearn.decomposition import PCA
from ClasificadorAudio import ClasificadorAudio
from ClasificadorImagen import ClasificadorImagen

class Evaluador:
    def __init__(self, modelo_path="modelos_entrenados.json", datos_procesados_path="datos_procesados.json"):
        print("Inicializando Evaluador...")
        self.caracteristicas_audio = None
        self.caracteristicas_imagen = None
        self.labels_imagen = None
        self.labels_audio = None

        print("Instanciando ClasificadorAudio y ClasificadorImagen...")
        self.clasificador_audio = ClasificadorAudio()
        self.clasificador_imagen = ClasificadorImagen()

        self.cargar_datos(modelo_path, datos_procesados_path)

    def cargar_datos(self, modelo_path, datos_procesados_path):
        """Carga las características de audio e imagen, etiquetas y centroides desde los archivos JSON."""
        print(f"Cargando datos desde {datos_procesados_path} y modelos desde {modelo_path}...")
        try:
            # Cargar datos procesados (características y etiquetas)
            with open(datos_procesados_path, "r") as f:
                datos_procesados = json.load(f)
                self.caracteristicas_audio = np.array(datos_procesados["audio"])
                self.caracteristicas_imagen = np.array(datos_procesados["imagen"])
                self.labels_audio = np.array(datos_procesados.get("etiquetas_audio", []))
                self.labels_imagen = np.array(datos_procesados.get("etiquetas_imagen", []))
            print("Datos procesados cargados correctamente.")

            # Cargar modelo entrenado (clasificadores)
            with open(modelo_path, "r") as f:
                modelos = json.load(f)
                
                # Cargar clasificador de audio
                audios_entrenamiento = modelos['clasificador_audio'].get('audios_entrenamiento', None)
                labels_audio_entrenamiento = modelos['clasificador_audio'].get('labels_audio_entrenamiento', None)
                if audios_entrenamiento is not None and labels_audio_entrenamiento is not None:
                    print("Cargando ClasificadorAudio desde modelos_entrenados.json...")
                    self.clasificador_audio.from_dict(modelos['clasificador_audio'])
                else:
                    print("Advertencia: No se encontraron datos de entrenamiento para el clasificador de audio.")
                
                # Cargar clasificador de imagen
                print("Cargando ClasificadorImagen desde modelos_entrenados.json...")
                self.clasificador_imagen.from_dict(modelos["clasificador_imagen"])
            
            print("Datos y modelos cargados exitosamente para evaluación.")
        except Exception as e:
            print(f"Error al cargar datos o modelos: {e}")

    def reducir_dimensionalidad(self, caracteristicas, n_componentes=3):
        """
        Aplica PCA para reducir la dimensionalidad de las características.

        :param caracteristicas: Array de características con forma (n_samples, n_features).
        :param n_componentes: Número de componentes principales para reducir.
        :return: Array transformado con forma (n_samples, n_componentes).
        """
        pca = PCA(n_components=n_componentes)
        caracteristicas_reducidas = pca.fit_transform(caracteristicas)
        print(f"Dimensionalidad reducida de {caracteristicas.shape[1]} a {n_componentes} componentes principales.")
        return caracteristicas_reducidas

    def calcular_estadisticas_por_etiqueta(self, caracteristicas, etiquetas, tipo):
        """
        Calcula las estadísticas por etiqueta (promedio y varianza) y las muestra en tablas separadas por tipo.

        :param caracteristicas: Array de características con forma (n_samples, n_features).
        :param etiquetas: Array de etiquetas correspondientes a cada muestra.
        :param tipo: Tipo de características ('audio' o 'imagen').
        """
        etiquetas_unicas = np.unique(etiquetas)
        
        # Mostrar encabezados separados para cada sección
        print(f"Estadísticas de {tipo.capitalize()}: Promedio")
        print(f"{'Etiqueta':<15} {'Promedio':<100}")
        print("=" * 120)
        for etiqueta in etiquetas_unicas:
            indices = np.where(etiquetas == etiqueta)
            caracteristicas_etiqueta = caracteristicas[indices]
            promedio = np.mean(caracteristicas_etiqueta, axis=0)
            promedio_str = np.array2string(promedio, precision=3, separator=', ', suppress_small=True, max_line_width=80)
            print(f"{etiqueta:<15} {promedio_str:<100}")
        print("\n" + "=" * 120 + "\n")

        print(f"Estadísticas de {tipo.capitalize()}: Varianza")
        print(f"{'Etiqueta':<15} {'Varianza':<100}")
        print("=" * 120)
        for etiqueta in etiquetas_unicas:
            indices = np.where(etiquetas == etiqueta)
            caracteristicas_etiqueta = caracteristicas[indices]
            varianza = np.var(caracteristicas_etiqueta, axis=0)
            varianza_str = np.array2string(varianza, precision=3, separator=', ', suppress_small=True, max_line_width=80)
            print(f"{etiqueta:<15} {varianza_str:<100}")
        print("\n" + "=" * 120 + "\n")

    def evaluar_audio(self):
        """Evalúa el rendimiento del clasificador de audio y calcula el porcentaje de aciertos."""
        if self.clasificador_audio is None or not hasattr(self.clasificador_audio, 'predecir'):
            print("Error: El clasificador de audio no está entrenado o no tiene el método 'predecir'.")
            return 0.0

        aciertos = 0
        total = len(self.caracteristicas_audio)
        for i, caracteristicas in enumerate(self.caracteristicas_audio):
            # Predicción real usando el clasificador
            prediccion = self.clasificador_audio.predecir(caracteristicas)
            if prediccion == self.labels_audio[i]:
                aciertos += 1
        porcentaje_aciertos = (aciertos / total) * 100 if total > 0 else 0.0
        print(f"Porcentaje de aciertos en audios: {porcentaje_aciertos:.2f}%")
        return porcentaje_aciertos

    def evaluar_imagen(self):
        """Evalúa el rendimiento del clasificador de imagen y calcula el porcentaje de aciertos."""
        if self.clasificador_imagen is None or not hasattr(self.clasificador_imagen, 'predecir'):
            print("Error: El clasificador de imagen no está entrenado o no tiene el método 'predecir'.")
            return 0.0

        aciertos = 0
        total = len(self.caracteristicas_imagen)
        for i, caracteristicas in enumerate(self.caracteristicas_imagen):
            # Predicción real usando el clasificador
            prediccion = self.clasificador_imagen.predecir(caracteristicas)
            if prediccion == self.labels_imagen[i]:
                aciertos += 1
        porcentaje_aciertos = (aciertos / total) * 100 if total > 0 else 0.0
        print(f"Porcentaje de aciertos en imágenes: {porcentaje_aciertos:.2f}%")
        return porcentaje_aciertos

    def ejecutar_evaluacion(self):
        print("Evaluando el rendimiento de los clasificadores...")
        aciertos_audio = self.evaluar_audio()
        aciertos_imagen = self.evaluar_imagen()
        print("Evaluación completada.")
        return aciertos_audio, aciertos_imagen

    def plot_3d_caracteristicas(self, tipo='audio'):
        if tipo == 'audio':
            caracteristicas = self.caracteristicas_audio
            etiquetas = self.labels_audio
            titulo = "Distribución 3D de Características de Audio"
        else:
            caracteristicas = self.caracteristicas_imagen
            etiquetas = self.labels_imagen
            titulo = "Distribución 3D de Características de Imagen"

        if caracteristicas is None or caracteristicas.shape[1] < 3:
            print(f"Advertencia: Las características de {tipo} no están disponibles o tienen menos de 3 dimensiones.")
            return

        if caracteristicas.shape[1] > 3:
            caracteristicas = self.reducir_dimensionalidad(caracteristicas)

        # Asignar colores a cada etiqueta
        etiquetas_unicas = np.unique(etiquetas)
        num_etiquetas = len(etiquetas_unicas)
        colores_cmap = plt.cm.get_cmap('tab10', num_etiquetas)

        # Crear un diccionario para mapear etiquetas a colores
        etiqueta_a_color = {etiqueta: colores_cmap(idx) for idx, etiqueta in enumerate(etiquetas_unicas)}

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Graficar cada punto con el color correspondiente a su etiqueta
        for etiqueta in etiquetas_unicas:
            indices = np.where(etiquetas == etiqueta)
            ax.scatter(
                caracteristicas[indices, 0], 
                caracteristicas[indices, 1], 
                caracteristicas[indices, 2],
                color=etiqueta_a_color[etiqueta],
                marker='o', alpha=0.7, s=30, label=f"Etiqueta: {etiqueta}"
            )

        # Verificar si es tipo 'imagen' y hay centroides disponibles
        if tipo == 'imagen' and self.clasificador_imagen.centroides is not None and self.clasificador_imagen.etiquetas_centroides is not None:
            unique_labels = self.clasificador_imagen.etiquetas_centroides
            num_centroides = len(self.clasificador_imagen.centroides)
            num_unique_labels = len(np.unique(unique_labels))

            if num_centroides != num_unique_labels:
                print("Advertencia: El número de centroides no coincide con el número de etiquetas únicas.")
                # Asignar un color predeterminado si no hay correspondencia
                centroid_colors = ['black'] * num_centroides
            else:
                # Asignar el color de la etiqueta correspondiente a cada centroide
                centroid_colors = [etiqueta_a_color[etiqueta] for etiqueta in unique_labels]

            # Reducir dimensionalidad de los centroides si es necesario
            if self.clasificador_imagen.centroides.shape[1] > 3:
                centroides_reducidos = self.reducir_dimensionalidad(self.clasificador_imagen.centroides)
            else:
                centroides_reducidos = self.clasificador_imagen.centroides

            # Graficar los centroides
            for idx, centroide in enumerate(centroides_reducidos):
                etiqueta = unique_labels[idx]
                ax.scatter(
                    centroide[0], centroide[1], centroide[2],
                    color=centroid_colors[idx],
                    marker='^', s=100, edgecolors='k',
                    label=f"Centroide: {etiqueta}"
                )

        # Evitar duplicación de etiquetas en la leyenda
        handles, labels = ax.get_legend_handles_labels()
        unique = list(dict(zip(labels, handles)).items())
        ax.legend([item[1] for item in unique], [item[0] for item in unique])

        ax.set_xlabel("Componente Principal 1")
        ax.set_ylabel("Componente Principal 2")
        ax.set_zlabel("Componente Principal 3")
        ax.set_title(titulo)
        plt.show()