import streamlit as st 
import os
import shutil
from datetime import datetime
from PIL import Image
import json
import yaml
from src.growth_stage import crew as growth_crew

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

# 1. Paths
UPLOAD_FOLDER = "data/uploads/growth_stage"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 3. User Interface
st.title("🌾 Wheat Growth Stage Detection")
st.markdown("Upload a wheat image to detect its growth stage and receive tailored advice.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    if st.button("Run Growth Stage Prediction"):
        try:
            # 4. Save image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"{timestamp}_{uploaded_file.name}"
            dest_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 5. Update config/tasks.yaml input
            for task in growth_crew.config.tasks.get("tasks", []):
                if task.get("name") == "PredictWheatStage":
                    task["input"]["image_path"] = dest_path

            # 6. Run the crew
            result = growth_crew.crew.kickoff()
            
            # 8. Display results
            if result:
                growth_stage = str(result.get("PredictWheatStage", {}).get("growth_stage", "Unknown"))
                advice = result.get("GiveWheatAdvice", {}).get("advice", "No advice available.")

                # Display final result
                st.success(f"🌱 **Predicted Growth Stage:** {growth_stage}")
                st.info(f"📋 **Advice:** {advice}")
            else:
                st.error("❌ No results returned from prediction.")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
                    st.markdown("### 📋 Practical advice for the farmer:")
                    st.markdown(advice)
            else:
                st.error("❌ results.json file not found after execution.")

        except Exception as e:
            st.error(f"❌ Error during execution: {e}")
