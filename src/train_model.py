import numpy as np
import pandas as pd
import os
import socket
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import mlflow
import mlflow.sklearn
import joblib

# ==================== CONFIGURATION INITIALE ====================
# Désactiver les avertissements
warnings.filterwarnings("ignore")

# Debug: Afficher la structure des fichiers
print("\n=== ENVIRONNEMENT ===")
print("Répertoire courant:", os.getcwd())
print("Contenu:", os.listdir())
print("Contenu src:", os.listdir('src') if os.path.exists('src') else "src non trouvé")
print("=====================\n")

# ==================== CHARGEMENT DES DONNÉES ====================
dataset_path = os.path.join("src", "data", "Consommation-de-carburant_data.csv")
print(f"Chemin du dataset: {dataset_path}")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Le fichier {dataset_path} n'existe pas. Vérifiez le chemin.")

donnees = pd.read_csv(dataset_path, sep=",")

# ==================== PRÉTRAITEMENT ====================
# Remplacer les '?' par NaN
donnees.replace('?', np.nan, inplace=True)

# Convertir les colonnes numériques
cols_numeric = ['mpg', 'weight', 'acceleration', 'displacement', 'cylinders', 'model year', 'horsepower']
for col in cols_numeric:
    donnees[col] = pd.to_numeric(donnees[col], errors='coerce')

# Gestion des valeurs manquantes
imputer = SimpleImputer(strategy='mean')
donnees[cols_numeric] = imputer.fit_transform(donnees[cols_numeric])

# ==================== PRÉPARATION DES DONNÉES ====================
y_regression = donnees['mpg']
X = donnees[['weight', 'acceleration', 'displacement', 'cylinders', 'model year', 'horsepower']]

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_regression, test_size=0.2, random_state=42)

# ==================== ENTRAÎNEMENT DU MODÈLE ====================
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Prédictions et évaluation
y_pred = rf_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Affichage des résultats
print(f"\n=== PERFORMANCES ===")
print(f"Erreur quadratique moyenne : {mse:.4f}")
print(f"Coefficient R² : {r2:.4f}")
print("===================\n")

# ==================== SAUVEGARDE LOCALE ====================
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

joblib.dump(rf_reg, os.path.join(models_dir, 'rf_reg_model.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
print("✅ Modèles sauvegardés localement")

# ==================== MLFLOW TRACKING ====================
def setup_mlflow():
    """Configure MLflow selon l'environnement"""
    if os.environ.get('CI') == 'true':  # Mode GitHub Actions
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        print("Configuration MLflow: Mode CI détecté (stockage local)")
    else:  # Mode local
        mlflow.set_tracking_uri("http://localhost:5000")
        print("Configuration MLflow: Mode local détecté (serveur MLflow)")
    
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "Prediction_Consommation_Carburant"))

# Log dans MLflow
setup_mlflow()

with mlflow.start_run():
    # Log des paramètres
    mlflow.log_params({
        "n_estimators": 100,
        "random_state": 42,
        "test_size": 0.2,
        "model_type": "RandomForestRegressor"
    })
    
    # Log des métriques
    mlflow.log_metrics({
        "mse": mse,
        "r2": r2
    })
    
    # Log des modèles
    mlflow.sklearn.log_model(rf_reg, "rf_reg_model")
    mlflow.sklearn.log_model(scaler, "scaler_model")
    
    # Log des artefacts
    mlflow.log_artifact(os.path.join(models_dir, 'rf_reg_model.pkl'))
    mlflow.log_artifact(os.path.join(models_dir, 'scaler.pkl'))
    
    print("\n✅ Suivi MLflow complété avec succès !")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# ==================== CHARGEMENT DES MODÈLES (EXEMPLE) ====================
# Décommenter pour tester le chargement
"""
try:
    rf_model = mlflow.sklearn.load_model(f"runs:/{mlflow.active_run().info.run_id}/rf_reg_model")
    scaler_model = mlflow.sklearn.load_model(f"runs:/{mlflow.active_run().info.run_id}/scaler_model")
    print("\n✅ Modèles chargés depuis MLflow avec succès !")
except Exception as e:
    print(f"\n⚠️ Erreur lors du chargement: {str(e)}")
"""