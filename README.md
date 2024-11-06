# Trabajo Final Inteligencia Artificial I – Visión Artificial y Reconocimiento de Voz

Este repositorio contiene el código desarrollado para el proyecto final de la materia Inteligencia Artificial I de la Universidad Nacional de Cuyo, Facultad de Ingeniería. El objetivo del proyecto es implementar un agente de software capaz de controlar un brazo robótico utilizando reconocimiento de voz y visión artificial. El sistema identifica comandos de voz para seleccionar una de cuatro verduras específicas (papa, zanahoria, camote y berenjena) y luego, mediante visión artificial, localiza dicha verdura en una cinta transportadora, dirigiendo al brazo robótico para su colocación en canastas asignadas.

## Resumen del Proyecto

El sistema está construido sobre los siguientes componentes principales:

1. **Clasificador de Audio**: Utiliza el algoritmo K-Nearest Neighbors (K-NN) para clasificar los comandos de voz. Las características del audio se extraen mediante Coeficientes Cepstrales de Frecuencia Mel (MFCC) y contraste espectral, seleccionando componentes específicos para optimizar la precisión.

2. **Clasificador de Imagen**: Implementa el algoritmo K-means para clasificar imágenes de las verduras en la cinta transportadora. Las características visuales utilizadas incluyen momentos de Hu y el color representativo de cada verdura.

3. **Servidor Flask**: La API creada en Flask permite que los usuarios envíen archivos de audio e imagen para clasificación. La API procesa los archivos, realiza predicciones y devuelve el resultado en tiempo real.

4. **Evaluador de Modelos**: Evalúa el desempeño de los clasificadores, calculando precisión y visualizando los datos en 3D para verificar la separación de clases y analizar la varianza y promedio de características por clase.

## Estructura del Repositorio

El repositorio contiene los siguientes archivos y carpetas:

- `ProcesadorAudio.py`: Clase para el procesamiento de audio, incluyendo carga, preprocesamiento y extracción de características.
- `ProcesadorImagen.py`: Clase para el procesamiento de imágenes, incluyendo carga, retoque LAB, eliminación de fondo y extracción de características.
- `ClasificadorAudio.py`: Clase que implementa el algoritmo K-NN para clasificar los comandos de voz.
- `ClasificadorImagen.py`: Clase que implementa el algoritmo K-means para clasificar las imágenes de verduras.
- `Entrenador.py`: Clase que gestiona el entrenamiento de los clasificadores, la carga de datos y el guardado de modelos en JSON.
- `Evaluador.py`: Clase que realiza la evaluación de los clasificadores y genera estadísticas de precisión, varianza y promedio de características.
- `servidor.py`: Servidor Flask que permite la clasificación de audio e imágenes a través de endpoints específicos.
- `saves/`: Carpeta donde se almacenan los modelos y datos de entrenamiento en formato JSON.

## Instalación

Para ejecutar el proyecto, siga estos pasos:

1. **Clonar el Repositorio**:
   ```bash
   git clone https://github.com/MatiasArmani/IA1-ArmaniMatias.git
   cd IA1-ArmaniMatias
   
## Instalación de Dependencias
Asegúrese de tener Python 3.7+ y ejecute el siguiente comando para instalar las bibliotecas necesarias:

```bash
pip install -r requirements.txt
