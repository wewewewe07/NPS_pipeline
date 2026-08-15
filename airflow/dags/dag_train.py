"""
DAG 2: dag_train
Train + so sánh model, chọn best, đánh giá test MAE, ghi model registry.
"""
import sys
sys.path.insert(0, "/opt/airflow/project")

import json
from datetime import datetime
import pandas as pd
import joblib
from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_DIR = "/opt/airflow/project/data"
MODELS_DIR = "/opt/airflow/project/models"
TARGET_COL = "target"


def load_gold(split):
    df = pd.read_csv(f"{DATA_DIR}/gold/{split}_gold.csv")
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


def task_train_and_select(**context):
    sys.path.insert(0, "/opt/airflow/project")
    from shared.model_training import compare_models, retrain_winner_on_train_dev

    X_train_raw, y_train = load_gold("train")
    X_dev_raw, y_dev = load_gold("dev")

    result = compare_models(X_train_raw, y_train, X_dev_raw, y_dev)

    print("So sánh model trên dev:")
    for name, params, _, mae, _ in result["results_sorted"]:
        print(f"  {name}: dev_mae={mae:.4f} params={params}")

    winner_name, winner_params, winner_model, winner_mae, winner_scale = result["winner"]
    print(f">>> Winner: {winner_name} dev_mae={winner_mae:.4f}")

    scaler = result["scaler"]
    if winner_scale != "stacking":
        print("[Retrain] Re-training winner trên train+dev combined...")
        retrained_model, retrained_scaler = retrain_winner_on_train_dev(
            winner_name, winner_params, X_train_raw, y_train, X_dev_raw, y_dev
        )
        winner_model = retrained_model
        scaler = retrained_scaler

    joblib.dump(winner_model, f"{MODELS_DIR}/best_model.joblib")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.joblib")

    meta = {
        "model_name": winner_name,
        "params": winner_params,
        "dev_mae": winner_mae,
        "needs_scaling": winner_scale == "scaled",
        "feature_columns": result["feature_columns"],
        "uses_feature_engineering": True,
    }
    with open(f"{MODELS_DIR}/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    context["ti"].xcom_push(key="model_meta", value=meta)


def task_evaluate_and_register(**context):
    sys.path.insert(0, "/opt/airflow/project")
    from shared.inference import load_model_bundle, predict_from_gold
    from shared.db import init_tables, write_model_registry
    from sklearn.metrics import mean_absolute_error

    df_test = pd.read_csv(f"{DATA_DIR}/gold/test_gold.csv")
    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL]

    bundle = load_model_bundle(MODELS_DIR)
    y_pred, _ = predict_from_gold(X_test, bundle)
    test_mae = mean_absolute_error(y_test, y_pred)
    print(f"Test MAE: {test_mae:.4f}")

    meta = bundle["meta"]
    meta["test_mae"] = test_mae
    with open(f"{MODELS_DIR}/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    init_tables()
    write_model_registry(meta["model_name"], meta["params"], meta["dev_mae"], test_mae)


default_args = {"owner": "tung", "retries": 1}

with DAG(
    dag_id="dag_train",
    description="Train + so sánh model (shared/model_training.py), chọn best, ghi model registry",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["nps", "train"],
) as dag:

    train_task = PythonOperator(task_id="train_and_select", python_callable=task_train_and_select)
    eval_task = PythonOperator(task_id="evaluate_and_register", python_callable=task_evaluate_and_register)

    train_task >> eval_task
