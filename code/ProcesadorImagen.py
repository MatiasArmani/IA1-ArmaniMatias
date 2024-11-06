import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class ProcesadorImagen:
    def __init__(self, ruta_imagen, lower_white=0, upper_white=255):
        self.ruta_imagen = ruta_imagen
        self.imagen = None
        self.imagen_retoque = None
        self.caracteristicas = None
        self.mascara_fondo = None
        self.imagen_sin_fondo = None
        self.lower_white = lower_white
        self.upper_white = upper_white
        self.errores_porcentaje_fondo = []
        self.imagen_contorno = None
        self.mascara_color = None

    def cargar_imagen(self):
        try:
            self.imagen = cv2.imread(self.ruta_imagen)
            if self.imagen is None:
                raise ValueError(f"No se pudo cargar la imagen desde la ruta: {self.ruta_imagen}")
            print(f"Imagen cargada correctamente desde {self.ruta_imagen}")
        except FileNotFoundError:
            print(f"Error: Archivo no encontrado: {self.ruta_imagen}")
            raise
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            raise

    def aplicar_retoque_lab(self):
        lab = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        self.imagen_retoque = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        print("Filtro LAB aplicado para mejorar la diferenciación.")

    def eliminar_fondo(self):
        if self.imagen_retoque is None:
            print("Error: La imagen con retoques no ha sido generada.")
            raise ValueError("La imagen con retoques no ha sido generada.")

        try:
            hsv = cv2.cvtColor(self.imagen_retoque, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([0, 0, self.lower_white])
            upper_bound = np.array([180, 30, self.upper_white])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
            self.mascara_fondo = mask
            mask_fg = cv2.bitwise_not(mask)
            self.imagen_sin_fondo = cv2.bitwise_and(self.imagen, self.imagen, mask=mask_fg)
            print("Fondo eliminado exitosamente.")

            porcentaje_fondo = np.sum(mask == 255) / mask.size
            if porcentaje_fondo < 0.5:
                print(f"Advertencia: Menos del 50% del fondo eliminado en {self.ruta_imagen}")
                self.errores_porcentaje_fondo.append(self.ruta_imagen)
        except Exception as e:
            print(f"Error al eliminar el fondo: {e}")
            raise

    def filtrar_outliers(self, colores):
        """Filtra los outliers usando el método del rango intercuartílico (IQR) para cada componente de color."""
        filtros = []
        for i in range(3):  # Iterar sobre los canales R, G y B
            Q1 = np.percentile(colores[:, i], 25)
            Q3 = np.percentile(colores[:, i], 75)
            IQR = Q3 - Q1
            filtro = (colores[:, i] >= (Q1 - 1.5 * IQR)) & (colores[:, i] <= (Q3 + 1.5 * IQR))
            filtros.append(filtro)
        
        # Crear una máscara que solo mantiene los píxeles que cumplen con todos los filtros
        filtro_total = np.logical_and.reduce(filtros)
        colores_filtrados = colores[filtro_total]
        
        return colores_filtrados

    def extraer_caracteristicas(self):
        if self.imagen_sin_fondo is None:
            print("Error: La imagen sin fondo no ha sido procesada.")
            raise ValueError("La imagen sin fondo no ha sido procesada.")

        try:
            gris = cv2.cvtColor(self.imagen_sin_fondo, cv2.COLOR_BGR2GRAY)
            contornos, _ = cv2.findContours(gris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contornos:
                contorno = max(contornos, key=cv2.contourArea)
                momentos = cv2.moments(contorno)
                hu = cv2.HuMoments(momentos).flatten()
                hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
                self.imagen_contorno = self.imagen_sin_fondo.copy()
                cv2.drawContours(self.imagen_contorno, [contorno], -1, (0, 255, 0), 2)

                # Crear una máscara a partir del contorno
                mascara = np.zeros_like(gris)
                cv2.drawContours(mascara, [contorno], -1, 255, -1)

                # Extraer píxeles dentro del contorno en el espacio RGB
                rgb = self.imagen_sin_fondo[mascara == 255]
                if rgb.size == 0:
                    color_representativo = np.array([0, 0, 0], dtype=np.uint8)
                    print("Advertencia: No se encontraron píxeles dentro del contorno para calcular el color.")
                else:
                    rgb_filtrado = self.filtrar_outliers(rgb)
                    if rgb_filtrado.size == 0:
                        print("Advertencia: No se encontraron colores válidos después del filtrado.")
                        color_representativo = np.array([0, 0, 0], dtype=np.uint8)
                    else:
                        color_representativo = np.median(rgb_filtrado, axis=0).astype(np.uint8)

                # Crear una imagen con el color representativo
                self.mascara_color = np.zeros_like(self.imagen_sin_fondo)
                self.mascara_color[mascara == 255] = color_representativo
            else:
                hu = np.zeros(7)
                self.imagen_contorno = self.imagen_sin_fondo.copy()
                print("Advertencia: No se encontraron contornos para calcular los momentos de Hu.")

            indices_hu_seleccionados = [0, 1, 2, 3]
            hu_seleccionados = hu[indices_hu_seleccionados]

            print(f"Color representativo (B, G, R): {color_representativo}")

            # Almacenar las características seleccionadas en un array de tipo float
            self.caracteristicas = np.concatenate((hu_seleccionados, color_representativo)).astype(np.float32)
            #self.caracteristicas = color_representativo.astype(np.float32)
            print(f"Características extraídas: {self.caracteristicas}")
            print(f"Número de características extraídas: {len(self.caracteristicas)}")
        except Exception as e:
            print(f"Error al extraer características: {e}")
            raise

    def visualizar_resultados(self):
        try:
            fig, axs = plt.subplots(1, 6, figsize=(24, 5))
            imagen_rgb = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB)
            axs[0].imshow(imagen_rgb)
            axs[0].set_title("Imagen Original")
            axs[0].axis('off')

            imagen_retoque_rgb = cv2.cvtColor(self.imagen_retoque, cv2.COLOR_BGR2RGB)
            axs[1].imshow(imagen_retoque_rgb)
            axs[1].set_title("Imagen con Retoques")
            axs[1].axis('off')

            axs[2].imshow(self.mascara_fondo, cmap='gray')
            axs[2].set_title("Máscara de Fondo")
            axs[2].axis('off')

            imagen_sin_fondo_rgb = cv2.cvtColor(self.imagen_sin_fondo, cv2.COLOR_BGR2RGB)
            axs[3].imshow(imagen_sin_fondo_rgb)
            axs[3].set_title("Imagen sin Fondo")
            axs[3].axis('off')

            imagen_contorno_rgb = cv2.cvtColor(self.imagen_contorno, cv2.COLOR_BGR2RGB)
            axs[4].imshow(imagen_contorno_rgb)
            axs[4].set_title("Imagen con Contornos")
            axs[4].axis('off')

            mascara_color_rgb = cv2.cvtColor(self.mascara_color, cv2.COLOR_BGR2RGB)
            axs[5].imshow(mascara_color_rgb)
            axs[5].set_title("Máscara con Color")
            axs[5].axis('off')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error al visualizar resultados: {e}")
            raise

def procesar_y_mostrar_imagen(ruta_imagen, lower_white=0, upper_white=255):
    procesador = ProcesadorImagen(ruta_imagen, lower_white=lower_white, upper_white=upper_white)
    try:
        procesador.cargar_imagen()
        procesador.aplicar_retoque_lab()
        procesador.eliminar_fondo()
        procesador.extraer_caracteristicas()
        print("Características extraídas de la imagen:")
        print(procesador.caracteristicas)
        procesador.visualizar_resultados()
    except Exception as e:
        print(f"Error al procesar la imagen {ruta_imagen}: {e}")

if __name__ == "__main__":
    ruta = input("Ingrese la ruta del archivo o carpeta para analizar: ").strip()
    if os.path.isfile(ruta):
        procesar_y_mostrar_imagen(ruta)
    elif os.path.isdir(ruta):
        archivos = [os.path.join(ruta, f) for f in os.listdir(ruta) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not archivos:
            print("No se encontraron imágenes en la carpeta especificada.")
        else:
            for ruta_imagen in archivos:
                procesar_y_mostrar_imagen(ruta_imagen)
    else:
        print("Error: La ruta no existe o no es válida. Verifica e intenta nuevamente.")