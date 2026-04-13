import streamlit as st
import shutil
import os
from src.harvest_disease import crew as harvest_crew
from datetime import datetime
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.detach(), encoding='utf-8')

# Set page configuration
st.set_page_config(
    page_title="Agricultural Image Crop Yield Prediction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for agricultural theme
st.markdown("""
<style>
    .main {
        background-color: #f5f5dc;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    h1, h2, h3 {
        color: #2e7d32;
    }
    /* Sidebar styling - fixed dark color */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #1e3a32 !important;
    }
    
</style>
""", unsafe_allow_html=True)

# Création du dossier d’images si non existant
UPLOAD_FOLDER = "data/uploads/harvest_disease"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("📊🌾 Crop Yield Prediction")
st.markdown("""
This app uses a deep learning model to detect the crop yield condition of wheat (e.g., healthy crop yield, poor crop yield) through image analysis and provides tailored agricultural advice based on the detected yield status""")

# Add a back button to return to home
if st.button("← Back to Home"):
    st.switch_page("app.py")

uploaded_file = st.file_uploader("Upload an image of a wheat ear ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Sauvegarder le fichier dans le dossier des images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{uploaded_file.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(file_path, caption="Uploaded Image", use_column_width=True)

    # Modifier les inputs de la tâche pour utiliser ce fichier
    for task in harvest_crew.config.tasks.get("tasks", []):
        if "image_paths" in task.get("input", {}):
            task["input"]["image_paths"] = [file_path]

    if st.button("🔍 Predict wheat crop forecasting"):
        result = harvest_crew.crew.kickoff()
        predictions = result.get("classify_harvest_disease_task", {}).get("predictions", [])
        enriched = result.get("solution_task", {}).get("predictions", [])

        # Afficher les résultats
        for pred in enriched:
            if pred['disease'] == "Wheat_healthy ":
                st.success(f"🌿 Wheat Condition: Healthy crop yield")
            else:
                st.warning(f"🌾 Wheat Condition: Poor crop yield")
            st.info(f"ℹ️ solution: {pred['solution']}")
