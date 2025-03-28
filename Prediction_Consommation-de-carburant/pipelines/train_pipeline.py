from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.train import train_model

default_args = {"start_date": datetime(2024, 1, 1)}
dag = DAG("train_pipeline", default_args=default_args, schedule_interval="0 0 * * *")

train_task = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)
train_task
