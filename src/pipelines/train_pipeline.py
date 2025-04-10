from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from train_model import train_model


# Arguments par défaut pour le DAG
default_args = {
    "start_date": datetime(2024, 1, 1),  # Date de début de l'exécution
    "retries": 1,  # Nombre de tentatives de réexécution en cas d'échec
}

# Création du DAG
dag = DAG(
    "train_pipeline", 
    default_args=default_args, 
    description="Un pipeline pour entraîner le modèle", 
    schedule_interval="0 0 * * *",  # À minuit chaque jour
    catchup=False  # Empêche l'exécution des anciennes tâches non exécutées
)

# Définition de la tâche train_task qui appelle la fonction train_model
train_task = PythonOperator(
    task_id="train_model", 
    python_callable=train_model, 
    dag=dag
)

# Si tu veux ajouter d'autres étapes, tu peux enchaîner ici avec d'autres tâches
# Exemple: task_2 = PythonOperator(...) 

train_task  # Il n'est pas nécessaire de l'afficher ici, mais c'est correct
