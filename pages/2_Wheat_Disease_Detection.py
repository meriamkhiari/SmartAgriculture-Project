import streamlit as st
import shutil
import os
import sys
import io
from datetime import datetime
from src.disease_detection import crew as disease_crew

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.detach(), encoding='utf-8')

# Set page configuration
st.set_page_config(
    page_title="Agricultural Image Wheat Disease Classification",
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

# Create upload directory if not exists
UPLOAD_FOLDER = "data/uploads/disease_detection"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("🌾 Wheat Disease Detection")

uploaded_file = st.file_uploader("Upload an image of a wheat leaf", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{uploaded_file.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(file_path, caption="Uploaded Image", use_column_width=True)

    # Update input paths in task config
    # Note: Accessing internal config directly is not ideal, but keeping it for now to match original logic
    for task in disease_crew.config.tasks.get("tasks", []):
        if "image_paths" in task.get("input", {}):
            task["input"]["image_paths"] = [file_path]

    if st.button("🔍 Classify Disease"):
        result = disease_crew.crew.kickoff()
        # ... rest of the code remains similar but needs path fixes for results
        predictions = result.get("classify_disease_task", {}).get("predictions", [])
        enriched = result.get("enrich_with_info_task", {}).get("predictions", [])

        # Display results
        for pred in enriched:
            st.success(f"🌿 Disease Detected: {pred['disease']}")
            st.info(f"ℹ️ Info: {pred['info']}")

        # Display YOLO annotated image if available
        uploaded_filename = os.path.basename(file_path)
        filename_base = os.path.splitext(uploaded_filename)[0]
        output_folder = "data/results"

        annotated_path = None
        if os.path.exists(output_folder):
            for fname in os.listdir(output_folder):
                if fname.startswith(f"annotated_{filename_base}"):
                    annotated_path = os.path.join(output_folder, fname)
                    break

        if annotated_path:
            st.image(annotated_path, caption="🧠 YOLO Annotated Result", use_column_width=True)
        else:
            st.warning("⚠️ No YOLO annotated image found.")
