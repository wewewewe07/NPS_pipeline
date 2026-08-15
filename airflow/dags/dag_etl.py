"""
DAG 1: dag_etl
Bronze -> Silver -> Gold, chạy khi có data mới (manual trigger hoặc khi bronze thay đổi).
"""
import sys
sys.path.insert(0, "/opt/airflow/project")

from datetime import datetime
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

from shared.preprocessing import bronze_to_silver, silver_to_gold, save_encoder, TARGET_COL

DATA_DIR = "/opt/airflow/project/data"
MODELS_DIR = "/opt/airflow/project/models"
RANDOM_STATE = 42


def nps_group(target: int) -> str:
    if target <= 6:
        return "detractor"
    elif target <= 8:
        return "passive"
    return "promoter"


def task_bronze_to_silver(**context):
    df_raw = pd.read_csv(f"{DATA_DIR}/bronze/nps_data2_id.csv")
    df_silver = bronze_to_silver(df_raw)
    df_silver.to_csv(f"{DATA_DIR}/silver/nps_silver.csv", index=False)
    print(f"Silver: {df_silver.shape}")


def task_split(**context):
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(f"{DATA_DIR}/silver/nps_silver.csv")
    strata = df["target"].apply(nps_group)

    df_train, df_temp, s_train, s_temp = train_test_split(
        df, strata, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
    )
    df_dev, df_test = train_test_split(
        df_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=s_temp
    )

    df_train.to_csv(f"{DATA_DIR}/splits/train.csv", index=False)
    df_dev.to_csv(f"{DATA_DIR}/splits/dev.csv", index=False)
    df_test.to_csv(f"{DATA_DIR}/splits/test.csv", index=False)
    print(f"train={len(df_train)} dev={len(df_dev)} test={len(df_test)}")


def task_build_gold(**context):
    df_train = pd.read_csv(f"{DATA_DIR}/splits/train.csv")
    df_dev = pd.read_csv(f"{DATA_DIR}/splits/dev.csv")
    df_test = pd.read_csv(f"{DATA_DIR}/splits/test.csv")

    X_train, encoder = silver_to_gold(df_train, fit_encoder=True)
    X_dev, _ = silver_to_gold(df_dev, encoder=encoder, fit_encoder=False)
    X_test, _ = silver_to_gold(df_test, encoder=encoder, fit_encoder=False)

    save_encoder(encoder, f"{MODELS_DIR}/encoder.joblib")

    for name, X, df in [("train", X_train, df_train), ("dev", X_dev, df_dev), ("test", X_test, df_test)]:
        gold = X.copy()
        gold[TARGET_COL] = df[TARGET_COL].values
        gold.to_csv(f"{DATA_DIR}/gold/{name}_gold.csv", index=False)


default_args = {"owner": "tung", "retries": 1}

with DAG(
    dag_id="dag_etl",
    description="Bronze -> Silver -> Gold + train/dev/test split cho NPS pipeline",
    default_args=default_args,
    schedule=None,  # manual trigger, hoặc đặt cron nếu bronze data cập nhật định kỳ
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["nps", "etl"],
) as dag:

    bronze_to_silver_task = PythonOperator(
        task_id="bronze_to_silver", python_callable=task_bronze_to_silver
    )
    split_task = PythonOperator(task_id="split_train_dev_test", python_callable=task_split)
    build_gold_task = PythonOperator(task_id="build_gold", python_callable=task_build_gold)

    bronze_to_silver_task >> split_task >> build_gold_task
