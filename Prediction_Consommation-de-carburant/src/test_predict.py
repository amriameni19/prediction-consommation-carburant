from predict_model import predict

# Exemple de caractéristiques d’un véhicule
sample_features = {
    'weight': 3000,         # poids
    'acceleration': 12.0,   # accélération
    'displacement': 200.0,  # cylindrée
    'cylinders': 4,         # nombre de cylindres
    'model_year': 76,       # année du modèle
    'horsepower': 90.0      # puissance
}

# Appel de la fonction
result = predict(sample_features)

# Affichage du résultat
print("Consommation réelle estimée (MPG) :", result['consommation_reelle'])
