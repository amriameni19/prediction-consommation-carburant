import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

def main():
    # Configuration MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Prediction_Carburant")
    
    with mlflow.start_run(run_name="Production"):
        # ========== CHARGEMENT ==========
        donnees = pd.read_csv("src/data/Consommation-de-carburant_data.csv")
        
        # ========== NETTOYAGE ==========
        colonnes = ['mpg', 'weight', 'acceleration', 'displacement', 'cylinders', 'model year', 'horsepower']
        donnees[colonnes] = donnees[colonnes].replace('?', np.nan)
        donnees[colonnes] = donnees[colonnes].apply(pd.to_numeric, errors='coerce')
        
        # ========== IMPUTATION ==========
        imputer = SimpleImputer(strategy='mean')
        donnees[colonnes] = imputer.fit_transform(donnees[colonnes])
        
        # ========== PREPARATION ==========
        X = donnees[['weight', 'acceleration', 'displacement', 'cylinders', 'model year', 'horsepower']]
        y = donnees['mpg']
        
        # ========== NORMALISATION ==========
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # ========== ENTRAINEMENT ==========
        modele = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=42
        )
        modele.fit(X_train, y_train)
        
        # ========== EVALUATION ==========
        predictions = modele.predict(X_test)
        metriques = {
            "mae": mean_absolute_error(y_test, predictions),
            "mse": mean_squared_error(y_test, predictions),
            "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
            "r2": r2_score(y_test, predictions)
        }
        
        # ========== ENREGISTREMENT ==========
        # Paramètres
        mlflow.log_params({
            "n_estimators": 150,
            "max_depth": 10,
            "taille_test": 0.2,
            "random_state": 42
        })
        
        # Métriques
        mlflow.log_metrics(metriques)
        
        # Sauvegarde des artefacts
        os.makedirs("models", exist_ok=True)
        joblib.dump(modele, "models/modele.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        
        # Log MLflow
        mlflow.sklearn.log_model(modele, "modele")
        mlflow.log_artifacts("models")
        
        print("✅ Entraînement terminé! Résultats disponibles dans MLflow")

if __name__ == "__main__":
    main()