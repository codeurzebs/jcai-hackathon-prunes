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
st.markdown("Charge une image de prune pour obtenir sa catégorie prédite par le modèle IA. \n \n Cette application est developpée par ZEBS HAUPUR & TIOJIO ROMAIN de la Communauté NGcodeX. \n \n Contact au +237692077005 (WhatsApp + Orange Money). \n Numero de compte Afriland First Bank: 08281371051. \n \n NB:Par defaut, les images qui ne sont pas des prunes seront classé comme pourrie.")

# Upload image
uploaded_file = st.file_uploader("📤 Choisis l image d une prune", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", use_column_width=True)

    if st.button("🔍 Prédire l image"):
        model = load_model()
        prediction = predict_image(model, image)
        st.success(f"✅ Prédiction IA : **{prediction.upper()}**")
