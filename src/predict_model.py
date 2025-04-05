import joblib
import numpy as np

# Charger les modèles sauvegardés
model_class = joblib.load('models/scaler.pkl')
model_reg = joblib.load('models/rf_reg_model.pkl')

def predict(features):
    # Convertir les caractéristiques du véhicule en un tableau numpy
    features_array = np.array([[
        features['weight'],
        features['acceleration'],
        features['displacement'],
        features['cylinders'],
        features['model_year'],
        features['origin'],
        features['horsepower']
    ]])
    
    # Prédire la classe (faible ou élevé)
    consumption_class = model_class.predict(features_array)[0]
    consumption_label = "Faible" if consumption_class == 1 else "Élevée"
    
    # Prédire la consommation de carburant réelle (en MPG)
    fuel_consumption = model_reg.predict(features_array)[0]
    
    # Retourner les deux résultats
    return {
        "consommation": consumption_label,
        "consommation_reelle": fuel_consumption
    }
