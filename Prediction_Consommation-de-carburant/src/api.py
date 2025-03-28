from fastapi import FastAPI
import joblib
import sys
import os

# Ajouter le chemin du projet dans sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



app = FastAPI()
model = joblib.load("models/best_model.pkl")

@app.post("/predict")
def predict(features: dict):
    prediction = model.predict([list(features.values())])
    return {"prediction": prediction.tolist()}
