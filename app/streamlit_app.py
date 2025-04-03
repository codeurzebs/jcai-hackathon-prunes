import sys
import torch
if hasattr(torch, 'classes'):
    del torch.classes
import sys
import os
sys.path.append(os.path.abspath("src"))
import streamlit as st
from PIL import Image
from predict_model import load_model, predict_image

st.set_page_config(page_title="Tri de Prunes 🍑", layout="centered")

st.title("🍑 JCIA Hackathon – Tri Automatique des Prunes")
st.markdown("Charge une image de prune pour obtenir sa catégorie prédite par le modèle IA. Cette application est developpée par ZEBS HAUPUR de la Communauté NGcodeX. contacter moi au +237692077005")

# Upload image
uploaded_file = st.file_uploader("📤 Choisis une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", use_column_width=True)

    if st.button("🔍 Prédire"):
        model = load_model()
        prediction = predict_image(model, image)
        st.success(f"✅ Prédiction : **{prediction.upper()}**")
