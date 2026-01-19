# Predicting Hit Songs using Spotify Data
- This project aims to predict the popularity of songs by analyzing various audio features from the Spotify dataset. Using multiple machine learning algorithms, I built a predictive model to determine whether a song will become a "hit" based on metrics like danceability, energy, and acousticness.

## Dataset
- Link: Kaggle - [Spotify Audio Features](https://www.kaggle.com/datasets/tomigelo/spotify-audio-features)
- File: `SpotifyAudioFeaturesApril2019.csv`

### Download
**Option 1: Using Kaggle CLI**
```bash
# Install kaggle CLI (if not installed)
pip install kaggle

# Download dataset (requires Kaggle API token)
kaggle datasets download -d tomigelo/spotify-audio-features
unzip spotify-audio-features.zip -d data/
```

**Option 2: Manual Download**
1. Visit [Kaggle Dataset Page](https://www.kaggle.com/datasets/tomigelo/spotify-audio-features)
2. Click "Download" button
3. Extract and place `SpotifyAudioFeaturesApril2019.csv` in `data/` directory

## Install
### Basic Requirements
> cat requirements.txt | xargs uv add
### Minimum Requirements (for deployment)
> pip install -r min_requirements.txt

## Project Structure
```
hitsong-classification/
├── notebook/
│   └── notebook.ipynb          # EDA, data preprocessing, model experimentation and hyperparameter tuning
├── data/
│   └── SpotifyAudioFeaturesApril2019.csv
├── train.py                    # Training script that preprocesses data, trains model, and saves to model.bin
├── fastapi_app.py              # FastAPI application for serving hit song predictions via REST API
├── flask_app.py                # Flask application for deployment (PythonAnywhere)
├── model.bin                   # Serialized trained model with scaler and metadata
├── Dockerfile                  # Docker configuration for containerized deployment
├── requirements.txt            # Python dependencies (for uv)
├── min_requirements.txt        # Minimum dependencies for deployment
├── pyproject.toml              # Project configuration
└── README.md
```

### Key Files
- **notebook.ipynb**: Jupyter notebook for exploratory data analysis, feature engineering, and model experimentation with multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting)
- **train.py**: Production training script that loads Spotify data, creates binary labels, applies feature scaling, trains the selected model, and exports to pickle format
- **fastapi_app.py**: FastAPI REST API server that loads the trained model and provides `/predict` endpoint for real-time hit song classification
- **flask_app.py**: Flask REST API server for deployment on WSGI platforms (e.g., PythonAnywhere)

## Run this project locally
### Jupyter 
```
# create env for jupyter
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=<project_name>
# run jupyter lab
uv run jupyter lab
```
### FastAPI
**Run with uvicorn (local)**
```bash
uv run uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

**Run with Docker**
```bash
docker build -t spotify-hit-api .

docker run --rm -p 8000:8000 spotify-hit-api
```

## Deployment

### Platform: PythonAnywhere
This application is deployed on **PythonAnywhere**, a cloud-based Python hosting platform that provides free hosting for Python web applications.

### Why Flask for Deployment?
While the local development uses FastAPI (`fastapi_app.py`), the production deployment uses Flask (`flask_app.py`) because PythonAnywhere currently doesn't support ASGI frameworks like FastAPI. Flask, being a WSGI framework, is fully supported on PythonAnywhere's infrastructure.

### Deployment Details
- **Platform**: PythonAnywhere (Free Tier)
- **Framework**: Flask (WSGI-compatible)
- **Base URL**: https://prograsshopper.pythonanywhere.com/
- **Server**: Waitress/WSGI server managed by PythonAnywhere

### Deployment Steps (PythonAnywhere)
1. **Upload Files**: Upload `flask_app.py`, `min_requirements.txt`, and `model.bin` to PythonAnywhere
   - Project Structure
    ```
    /home/<user_name>/
    ├── flask_app.py
    ├── model.bin
    ├── min_requirements.txt
    ```

2. **Install Dependencies**: Use PythonAnywhere's bash console to install required packages
   ```bash
   virtualenv --python=python3.11 <venv_name>
   source <venv_name>/bin/activate
   pip install -r min_requirements.txt 
   ```
3. **Configure WSGI**: Set up the WSGI configuration file to point to `flask_app.py`
4. **Reload Web App**: Reload the web application from the PythonAnywhere dashboard

### API Endpoints

#### 1. Health Check / API Information
**Endpoint**: `GET /`

**URL**: https://prograsshopper.pythonanywhere.com/

**Response**:
```json
{
  "message": "Hit Song Classification API (Flask)",
  "model_metrics": {
    "accuracy": 0.94,
    "f1": 0.11,
    "precision": 0.07,
    "recall": 0.23
  },
  "hit_threshold": 70,
  "scaling": true,
  "required_features": [...]
}
```

#### 2. Predict Hit Song
**Endpoint**: `POST /predict`

**URL**: https://prograsshopper.pythonanywhere.com/predict

**Request Body**:
```json
{
  "features": {
    "acousticness": 0.25,
    "danceability": 0.654,
    "duration_ms": 207857,
    "energy": 0.547,
    "instrumentalness": 0.0,
    "key": 10,
    "liveness": 0.0961,
    "loudness": -6.598,
    "mode": 1,
    "speechiness": 0.127,
    "tempo": 173.981,
    "time_signature": 4,
    "valence": 0.507
  }
}
```

**Response**:
```json
{
  "hit": 1,
  "confidence": 0.85,
  "probabilities": {
    "non_hit": 0.15,
    "hit": 0.85
  }
}
```

### Testing the Deployed API

**Using curl**:
```bash
curl -X POST https://prograsshopper.pythonanywhere.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "acousticness": 0.25,
      "danceability": 0.654,
      "duration_ms": 207857,
      "energy": 0.547,
      "instrumentalness": 0.0,
      "key": 10,
      "liveness": 0.0961,
      "loudness": -6.598,
      "mode": 1,
      "speechiness": 0.127,
      "tempo": 173.981,
      "time_signature": 4,
      "valence": 0.507
    }
  }'
```

**Using Python requests**:
```python
import requests

url = "https://prograsshopper.pythonanywhere.com/predict"
data = {
    "features": {
        "acousticness": 0.25,
        "danceability": 0.654,
        "duration_ms": 207857,
        "energy": 0.547,
        "instrumentalness": 0.0,
        "key": 10,
        "liveness": 0.0961,
        "loudness": -6.598,
        "mode": 1,
        "speechiness": 0.127,
        "tempo": 173.981,
        "time_signature": 4,
        "valence": 0.507
    }
}

response = requests.post(url, json=data)
print(response.json())
```
