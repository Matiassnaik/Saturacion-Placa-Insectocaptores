import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Análisis de Placa Insectocaptor", layout="wide")
st.title("🪰 Análisis de Saturación de Placas")
st.markdown("Subí una imagen de la placa y el sistema calculará la saturación total. Si está vacía, se notificará.")

archivo = st.file_uploader("📷 Subí una imagen de la placa", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen_pil = Image.open(archivo).convert("RGB")
    imagen = np.array(imagen_pil)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    imagen = cv2.resize(imagen, (800, 1200))

    alto, ancho = imagen.shape[:2]
    y1, y2 = int(alto * 0.1), int(alto * 0.9)
    x1, x2 = int(ancho * 0.1), int(ancho * 0.9)
    zona_util = imagen[y1:y2, x1:x2]

    # Procesamiento por 3 canales
    gris = cv2.cvtColor(zona_util, cv2.COLOR_BGR2GRAY)
    gris = cv2.equalizeHist(gris)
    _, bin_gris = cv2.threshold(gris, 130, 255, cv2.THRESH_BINARY_INV)

    hsv = cv2.cvtColor(zona_util, cv2.COLOR_BGR2HSV)
    canal_v = hsv[:, :, 2]
    _, bin_v = cv2.threshold(canal_v, 140, 255, cv2.THRESH_BINARY_INV)

    lab = cv2.cvtColor(zona_util, cv2.COLOR_BGR2Lab)
    canal_l = lab[:, :, 0]
    _, bin_l = cv2.threshold(canal_l, 140, 255, cv2.THRESH_BINARY_INV)

    # Fusión de máscaras
    binaria_comb = cv2.bitwise_or(bin_gris, bin_v)
    binaria_comb = cv2.bitwise_or(binaria_comb, bin_l)

    # Filtro para ignorar las líneas (consideradas muy delgadas)
    kernel = np.ones((2, 2), np.uint8)
    binaria_comb = cv2.morphologyEx(binaria_comb, cv2.MORPH_OPEN, kernel)

    # División en celdas
    filas, columnas = 12, 10
    h, w = binaria_comb.shape
    alto_celda = h // filas
    ancho_celda = w // columnas

    saturaciones = np.zeros((filas, columnas))
    suma_total_pixeles = 0
    suma_total_insectos = 0
    imagen_grilla = zona_util.copy()

    for i in range(filas):
        for j in range(columnas):
            y_ini = i * alto_celda
            y_fin = (i + 1) * alto_celda
            x_ini = j * ancho_celda
            x_fin = (j + 1) * ancho_celda

            celda = binaria_comb[y_ini:y_fin, x_ini:x_fin]
            pixeles_insectos = np.sum(celda == 255)
            pixeles_totales = celda.size

            saturacion = (pixeles_insectos / pixeles_totales) * 100
            saturaciones[i, j] = saturacion

            suma_total_insectos += pixeles_insectos
            suma_total_pixeles += pixeles_totales

            cv2.rectangle(imagen_grilla, (x_ini, y_ini), (x_fin, y_fin), (0, 255, 0), 1)

    saturacion_total = (suma_total_insectos / suma_total_pixeles) * 100

    # Mostrar resultado
    if saturacion_total < 10:
        st.markdown("### ⚪ Resultado: **Placa vacía o sin capturas significativas**")
    else:
        st.markdown(f"### 📊 Saturación Total Estimada: **{saturacion_total:.2f}%**")

    # Mostrar visuales
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(cv2.cvtColor(imagen_grilla, cv2.COLOR_BGR2RGB), caption="📸 Imagen con Grilla", use_container_width=True)

    with col2:
        st.image(binaria_comb, caption="⚫ Máscara Binaria Combinada", use_container_width=True, channels="GRAY")

    with col3:
        fig, ax = plt.subplots()
        heatmap = ax.imshow(saturaciones, cmap='hot', interpolation='nearest')
        ax.set_title("🔥 Mapa de Calor por Celda")
        plt.colorbar(heatmap, ax=ax, label="Saturación %")
        ax.set_xticks(np.arange(columnas))
        ax.set_yticks(np.arange(filas))
        ax.invert_yaxis()
        st.pyplot(fig)
