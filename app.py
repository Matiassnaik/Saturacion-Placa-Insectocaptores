import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

st.set_page_config(page_title="Análisis de Placa Insectocaptor", layout="wide")
st.title("🪰 Análisis de Saturación de Placas")
st.markdown("Subí una imagen. Si es muy similar a la placa vacía de referencia, será detectada como vacía.")

@st.cache_data
def cargar_referencia():
    referencia = cv2.imread("placa_vacia.jpg")  # Asegurate de tener esta imagen en tu carpeta
    return cv2.resize(referencia, (800, 1200))

ref_vacia = cargar_referencia()

archivo = st.file_uploader("📷 Subí una imagen de la placa", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen_pil = Image.open(archivo).convert("RGB")
    imagen = np.array(imagen_pil)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    imagen = cv2.resize(imagen, (800, 1200))

    gray_ref = cv2.cvtColor(ref_vacia, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    indice_ssim, _ = ssim(gray_ref, gray_img, full=True)

    if indice_ssim > 0.40:  # 🟢 Umbral bajo para tolerar reflejos
        st.markdown(f"### ⚪ Resultado: **Placa vacía o sin insectos visibles (SSIM: {indice_ssim:.2f})**")
        st.image(imagen_pil, caption="📸 Placa cargada (detectada como vacía)", use_container_width=True)
    else:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        gris = cv2.equalizeHist(gris)
        _, binaria = cv2.threshold(gris, 105, 255, cv2.THRESH_BINARY_INV)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        pixeles_insectos = np.sum(binaria == 255)
        total_pixeles = binaria.size
        saturacion_total = (pixeles_insectos / total_pixeles) * 100

        st.markdown(f"### 📊 Saturación Total Estimada: **{saturacion_total:.2f}%** (SSIM: {indice_ssim:.2f})")

        col1, col2 = st.columns(2)
        with col1:
            st.image(imagen_pil, caption="📸 Imagen cargada", use_container_width=True)
        with col2:
            st.image(binaria, caption="⚫ Máscara Binaria (insectos detectados)", use_container_width=True, channels="GRAY")
