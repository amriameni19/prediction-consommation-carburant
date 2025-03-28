import os

# Définition des dossiers et fichiers à créer
folders = ["notebooks", "src", "models", "pipelines", "tests", "config"]
files = ["README.md", "requirements.txt"]

# Création des dossiers
for folder in folders:
    os.makedirs(f"Prediction_Consommation-de-carburant/{folder}", exist_ok=True)

# Création des fichiers
for file in files:
    open(f"Prediction_Consommation-de-carburant/{file}", "w").close()

print("✅ Structure du projet créée avec succès ! 🎯")
