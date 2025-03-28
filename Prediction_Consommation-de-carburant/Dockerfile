# Utiliser une image de base Python
FROM python:3.9

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier tous les fichiers du projet dans le conteneur
COPY . /app
# Copier le fichier prometheus.yml dans le conteneur
COPY prometheus.yml /etc/prometheus/prometheus.yml

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port 8000 pour l'API
EXPOSE 8000

# Définir la commande pour lancer l'API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
