from flask import Flask, request, jsonify
from predict_model import predict  # Assurez-vous que le chemin vers votre fonction de prédiction est correct

app = Flask(__name__)

# Fonction pour valider et traiter les données d'entrée
def validate_car_features(data):
    required_fields = ["weight", "acceleration", "displacement", "cylinders", "model_year", "origin", "horsepower"]
    # Vérifier que toutes les clés nécessaires sont présentes
    for field in required_fields:
        if field not in data:
            return f"Le champ {field} est requis.", 400  # Erreur 400 si une clé manque
    return None, None

# Route pour la racine ("/")
@app.route("/", methods=["GET"])
def read_root():
    return jsonify({"message": "Bienvenue sur l'API de prédiction de consommation de carburant!"})

# Route pour effectuer une prédiction
@app.route("/predict/", methods=["POST"])
def predict_car():
    # Obtenir les données JSON de la requête
    data = request.get_json()
    
    # Valider les données d'entrée
    error_message, status_code = validate_car_features(data)
    if error_message:
        return jsonify({"error": error_message}), status_code
    
    # Appeler la fonction de prédiction
    prediction = predict(data)
    
    # Retourner les résultats de la prédiction
    return jsonify(prediction)

# Lancer l'application Flask
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
