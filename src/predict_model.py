import joblib
import numpy as np
import os
import pandas as pd

# Obtenir le chemin absolu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rf_reg_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# Charger les modèles
try:
    model_reg = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise Exception(f"Erreur lors du chargement des modèles: {str(e)}")

def predict(features):
    try:
        features_df = pd.DataFrame([[ 
            features['weight'],
            features['acceleration'],
            features['displacement'],
            features['cylinders'],
            features['model_year'],
            features['horsepower']
        ]],columns=['weight', 'acceleration', 'displacement', 'cylinders', 'model year', 'horsepower'])

        # Standardisation des features

        features_scaled = scaler.transform(features_df)

        # Prédiction
        fuel_consumption_mpg = model_reg.predict(features_scaled)[0]

       # Conversion de MPG à L/100km
        fuel_consumption_l_100km = 235.215 / fuel_consumption_mpg

        return {
            "consommation_reelle_mpg": float(fuel_consumption_mpg),
            "consommation_reelle_l_100km": float(fuel_consumption_l_100km),
            "unite_mpg": "MPG",
            "unite_l_100km": "L/100km",
            "commentaire_conversion": f"Conversion : 235.215 / {fuel_consumption_mpg} MPG = {fuel_consumption_l_100km:.2f} L/100km"
        }
    except Exception as e:
        raise Exception(f"Erreur lors de la prédiction: {str(e)}")
