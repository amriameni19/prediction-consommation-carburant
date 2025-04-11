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
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Fuel_Consumption_Prediction")
    
    with mlflow.start_run(run_name="RF_Regression"):
        # Chargement des données
        dataset_path = "src/data/Consommation-de-carburant_data.csv"
        donnees = pd.read_csv(dataset_path, sep=",")
        
        # Préprocessing
        donnees.replace('?', np.nan, inplace=True)
        cols_numeric = ['mpg', 'weight', 'acceleration', 'displacement', 'cylinders', 'model year', 'horsepower']
        for col in cols_numeric:
            donnees[col] = pd.to_numeric(donnees[col], errors='coerce')
        
        imputer = SimpleImputer(strategy='mean')
        donnees[cols_numeric] = imputer.fit_transform(donnees[cols_numeric])
        
        # Séparation des données
        y = donnees['mpg']
        X = donnees[['weight', 'acceleration', 'displacement', 'cylinders', 'model year', 'horsepower']]
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Entraînement
        rf_reg = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
        rf_reg.fit(X_train, y_train)
        
        # Évaluation
        y_pred = rf_reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log des métriques
        mlflow.log_metrics({
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })
        
        # Log des paramètres
        mlflow.log_params({
            "n_estimators": 150,
            "max_depth": 10,
            "test_size": 0.2,
            "random_state": 42
        })
        
        # Sauvegarde des artefacts
        os.makedirs("models", exist_ok=True)
        joblib.dump(rf_reg, "models/rf_reg_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        
        # Log des modèles et artefacts
        mlflow.sklearn.log_model(rf_reg, "model")
        mlflow.log_artifacts("models")
        
        print(f"Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()