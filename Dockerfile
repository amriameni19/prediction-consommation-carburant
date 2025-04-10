# Utiliser l'image de base Python
FROM python:3.10-slim

ENV MLFLOW_HOME=/mlflow \
    MLFLOW_VERSION=2.5.0

RUN apt-get update && apt-get install -y gcc libpq-dev curl \
    && pip install --upgrade pip \
    && pip install mlflow==${MLFLOW_VERSION} psycopg2-binary \
    && apt-get remove -y gcc \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p $MLFLOW_HOME
WORKDIR $MLFLOW_HOME

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
