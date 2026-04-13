# Agricultural Analysis Platform

A comprehensive agricultural analysis platform designed to assist farmers and agricultural scientists in monitoring crop health, growth stages, and field conditions using advanced AI models and satellite imagery.

## Features

### Satellite Image Segmentation
Utilizes deep learning models to segment agricultural satellite imagery, identifying different crop types, soil conditions, and dense vegetation areas.

### Wheat Disease Detection
Detects and classifies common wheat diseases from leaf images, including:
- Wheat Brown Rust
- Wheat Healthy
- Wheat Smut
- Wheat Yellow Rust
- Wheat Stem Rust

### Wheat Growth Stage Detection
Monitors the development of wheat crops through various growth stages (Filling, Flowering, Ripening) and provides tailored agricultural advice.

### Crop Yield Prediction
Analyzes wheat ear images to forecast crop yield conditions, distinguishing between healthy and poor yield prospects.

### Weed Detection
Identifies and classifies weeds in agricultural fields, providing management recommendations based on best practices.

### Insect Detection
Classifies various insects found in agricultural settings and provides detailed information on pest management.

## Technical Stack

- **Frontend**: Streamlit
- **AI Frameworks**: TensorFlow, PyTorch, Ultralytics (YOLO)
- **Agent Orchestration**: CrewAI
- **Data Processing**: NumPy, OpenCV, Pandas, Pillow
- **Search Integration**: DuckDuckGo Search API
- **LLM Integration**: Groq (Llama models)

## Project Structure

```
agriculture_platform/
├── app.py              # Main application entry point
├── data/               # User uploads and analysis results
├── models/             # Pre-trained AI models
├── pages/              # Streamlit page modules
├── src/                # Core logic and AI agents
│   ├── common/         # Shared utilities
│   ├── disease_detection/
│   ├── growth_stage/
│   ├── harvest_disease/
│   ├── insect_detection/
│   └── weed_detection/
├── .env.example        # Template for environment variables
├── .gitignore          # Git ignore configuration
└── requirements.txt    # Project dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/meriamkhiari/SmartAgriculture-Project.git
   cd SmartAgriculture-Project
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys (GROQ_API_KEY, etc.)

## Usage

Start the Streamlit application:
```bash
streamlit run app.py
```

Navigate through the sidebar to access different analysis tools.

## Production Considerations

- **Model Storage**: Ensure all `.h5` and `.pth` model files are placed in the `models/` directory.
- **Data Persistence**: Analysis results are stored in `data/results/`.
- **API Limits**: Be mindful of API rate limits for Groq and Search services.
- **Environment**: Use a proper production environment for hosting Streamlit (e.g., Streamlit Cloud, Heroku, or a dedicated VPS).

## License

This project is licensed under the terms provided by the Data Brains team at ESPRIT.
