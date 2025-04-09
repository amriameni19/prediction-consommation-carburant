from flask import Flask, request, jsonify
from predict_model import predict

app = Flask(__name__)

# Fonction de validation des données entrantes
def validate_car_features(data):
    required_fields = ["weight", "acceleration", "displacement", "cylinders", "model_year", "horsepower"]
    for field in required_fields:
        if field not in data:
            return f"Le champ {field} est requis.", 400
    return None, None

# Route d'accueil
@app.route("/", methods=["GET"])
def read_root():
    return jsonify({"message": "Bienvenue sur l'API de prédiction de consommation de carburant!"})

# Route pour la prédiction
@app.route("/predict/", methods=["POST"])
def predict_car():
    data = request.get_json()

    # Vérifier les données
    error_message, status_code = validate_car_features(data)
    if error_message:
        return jsonify({"error": error_message}), status_code

    try:
        prediction = predict(data)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Lancer l'application Flask
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
