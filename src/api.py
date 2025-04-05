from fastapi import FastAPI
from pydantic import BaseModel
from src.predict_model import predict  # Importer la fonction de prédiction

# Créer un modèle Pydantic pour valider les données d'entrée
class CarFeatures(BaseModel):
    weight: float
    acceleration: float
    displacement: float
    cylinders: int
    model_year: int
    origin: int
    horsepower: float

# Créer l'application FastAPI
app = FastAPI()

# Route pour effectuer une prédiction
@app.post("/predict/")
def predict_car(features: CarFeatures):
    # Convertir les données d'entrée en dictionnaire
    features_dict = features.dict()
    
    # Appeler la fonction de prédiction
    prediction = predict(features_dict)
    
    # Retourner les résultats de la prédiction
    return prediction

# Pour exécuter l'API avec uvicorn (si exécuté directement en ligne de commande)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
