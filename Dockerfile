# Utiliser l'image de base Python
FROM python:3.8-slim

# Définir les variables d'environnement
ENV MLFLOW_HOME=/mlflow
ENV MLFLOW_VERSION=2.6.1

# Installer les dépendances de MLflow
RUN pip install --upgrade pip \
    && pip install mlflow==2.5.0 \
    && pip install psycopg2-binary


# Créer le répertoire pour l'application
RUN mkdir -p $MLFLOW_HOME

# Exposer le port de MLflow
EXPOSE 5000

# Définir le répertoire de travail
WORKDIR $MLFLOW_HOME

# Lancer MLflow en mode serveur
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]                                                                         
