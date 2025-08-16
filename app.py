import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --- Configuration page ---
st.set_page_config(
    page_title="Détection de la Tuberculose - CNN",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Charger le modèle ---
@st.cache_resource
def charger_modele():
    return tf.keras.models.load_model("tuberculosis_model.keras")  # ton modèle entraîné

model = charger_modele()

# --- CSS design ---
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 1.5rem;
    }
    .positive {
        background-color: #fdecea;
        border-left: 6px solid #dc3545;
    }
    .negative {
        background-color: #e0f7e9;
        border-left: 6px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🫁 Détection de la Tuberculose avec CNN</h1>
    <p>Analysez une radiographie pulmonaire pour détecter la présence de Tuberculose</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ℹ️ À propos")
    st.success("""
    🧬 Cet outil repose sur un **réseau de neurones convolutionnel (CNN)**  
    entraîné sur des radiographies thoraciques.
    """)
    st.markdown("""
    **⚙️ Fonctionnement :**
    - 📥 **Entrée** : Image d'une radiographie pulmonaire (JPG/PNG)  
    - 📊 **Sortie** : Probabilité d'infection
    """)
    st.info("📈 **Précision du modèle : ~96%**")
    st.warning("""
    ⚠️ **Important :**  
    Cet outil est à objectif **éducatif** et **ne remplace pas** un diagnostic médical professionnel.
    """)

# --- Upload image ---
uploaded_file = st.file_uploader("📤 Choisissez une radiographie pulmonaire...", type=["jpg", "jpeg", "png"])

# --- Container pour les résultats ---
result_container = st.container()

if uploaded_file:
    result_container.empty()
    
    # Affichage image
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(img, caption="Radiographie chargée", use_container_width=True)

    # Prétraitement pour le modèle
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction
    prediction = model.predict(img_array)
    proba = prediction[0][0]

    # Déterminer si tuberculose
    is_tb = proba > 0.5  # modèle entraîné avec sigmoid (1 = TB, 0 = Normal)

    # Définir label et recommandations
    if is_tb:
        label = "Tuberculose détectée"
        conseil = "⚠️ Recommandation : consulter un pneumologue pour une analyse complémentaire."
        style_box = "positive"
        couleur_barre = "#dc3545"
        proba_affiche = proba
    else:
        label = "Poumons normaux"
        conseil = "✅ Recommandation : aucun signe de tuberculose détecté."
        style_box = "negative"
        couleur_barre = "#28a745"
        proba_affiche = 1 - proba

    # Affichage du résultat
    with result_container:
        st.markdown(f"""
        <div class="result-box {style_box}">
            <h2>{'🫁 ' + label}</h2>
            <p><strong>Probabilité :</strong> {proba_affiche:.2%}</p>
            <div style="background-color: #e9ecef; border-radius: 25px; height: 20px; overflow: hidden; margin-top: 1rem;">
                <div style="width: {proba_affiche*100}%; height: 100%; background-color: {couleur_barre}; border-radius: 25px;"></div>
            </div>
            <p style="margin-top: 1rem; font-weight: bold;">{conseil}</p>
            <br><br>
        </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 20px; margin-top: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h4 style="color: #495057; margin-bottom: 1rem;">🫁 Votre Assistant Détection Tuberculose</h4>
    <p style="font-size: 1em; color: #6c757d; margin-bottom: 0.5rem;">
        Créé avec passion par <strong>Youssouf</strong> pour vous aider à analyser les radiographies thoraciques.
    </p>
    <p style="font-size: 0.9em; color: #6c757d; margin-bottom: 1rem;">
        Version 2024 - Mis à jour régulièrement pour améliorer la précision
    </p>
    <div style="border-top: 1px solid #dee2e6; padding-top: 1rem;">
        <p style="font-size: 0.85em; color: #6c757d; font-style: italic;">
            ⚠️ Rappel important : Cet outil est éducatif et ne remplace pas un diagnostic médical professionnel.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
