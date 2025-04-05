import joblib
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import mlflow
import mlflow.sklearn

# Désactiver les avertissements
warnings.filterwarnings("ignore")

# Charger les données
dataset_path = "data/Consommation-de-carburant_data.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Le fichier {dataset_path} n'existe pas. Vérifiez le chemin.")

donnees = pd.read_csv(dataset_path, sep=",")

# Remplacer les '?' par NaN
donnees.replace('?', np.nan, inplace=True)

# Convertir les colonnes numériques
cols_numeric = ['mpg', 'weight', 'acceleration', 'displacement', 'cylinders', 'model year', 'horsepower']
for col in cols_numeric:
    donnees[col] = pd.to_numeric(donnees[col], errors='coerce')

# Gestion des valeurs manquantes
imputer = SimpleImputer(strategy='mean')
donnees[cols_numeric] = imputer.fit_transform(donnees[cols_numeric])

# Créer la variable cible de régression (mpg)
y_regression = donnees['mpg']  # Cible pour la régression
X = donnees[['weight', 'acceleration', 'displacement', 'cylinders', 'model year', 'horsepower']]  # Variables prédictives

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_regression, test_size=0.2, random_state=42)

# Entraînement du modèle Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Prédictions et évaluation
y_pred = rf_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Afficher les résultats
print(f"Erreur quadratique moyenne pour Random Forest (régression) : {mse:.4f}")
print(f"Coefficient de détermination R² pour Random Forest : {r2:.4f}")

# Création du dossier de sauvegarde des modèles
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Sauvegarde des modèles avec joblib (optionnel si MLflow gère déjà cela)
joblib.dump(rf_reg, os.path.join(models_dir, 'rf_reg_model.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))


print("\nLe modèle a été sauvegardé avec succès.")

# 1️⃣ Définir l'URI de tracking (MLflow UI)
mlflow.set_tracking_uri("http://localhost:5000")

# 2️⃣ Créer une expérience
mlflow.set_experiment("Prediction_Consommation_Carburant")

# 3️⃣ Démarrer un run
with mlflow.start_run():
    # Log des hyperparamètres (si besoin)
    mlflow.log_param("n_estimators", 100)  # Exemple pour RandomForest
    mlflow.log_param("max_depth", 10)

    # 🎯 Log du modèle dans MLflow
    mlflow.sklearn.log_model(rf_reg, "rf_reg_model")
    mlflow.sklearn.log_model(scaler, "scaler_model")

    # 🔹 Enregistrer les fichiers comme artefacts
    mlflow.log_artifact(os.path.join(models_dir, 'rf_reg_model.pkl'))
    mlflow.log_artifact(os.path.join(models_dir, 'scaler.pkl'))

    print("✅ Modèles et artefacts enregistrés avec succès dans MLflow ! 🚀")


import mlflow.sklearn

# Charger le modèle RandomForest
rf_model = mlflow.sklearn.load_model("runs:/ba28e042d2634fb7adb13b83da0ff4c7/rf_reg_model")

# Charger le scaler
scaler_model = mlflow.sklearn.load_model("runs:/ba28e042d2634fb7adb13b83da0ff4c7/scaler_model")

print("✅ Modèles chargés avec succès !")
