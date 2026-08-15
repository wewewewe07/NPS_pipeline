"""
DAG 3: dag_daily_predict
Vì dataset gốc không có cột ngày thật, mình GIẢ LẬP: rải 100 dòng test set ra N ngày
(mỗi ngày một batch nhỏ), data đến mỗi ngày để mô phỏng production.

Chạy schedule=@daily. Mỗi lần chạy (ứng với 1 ngày giả lập) sẽ:
1. Lấy batch của "ngày đó" (dựa vào sim_date đã gán sẵn, xem task_assign_sim_dates)
2. Predict bằng model mới nhất (best_model.joblib)
3. Ghi kết quả (kèm actual Target có sẵn trong test set) vào bảng predictions
   -> dashboard dùng để vẽ MAE/drift theo thời gian
"""
import sys
sys.path.insert(0, "/opt/airflow/project")

import json
from datetime import datetime, timedelta
import pandas as pd
import joblib
from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_DIR = "/opt/airflow/project/data"
MODELS_DIR = "/opt/airflow/project/models"
TARGET_COL = "target"
SIM_START_DATE = datetime(2026, 1, 1)  # ngày "giả lập" bắt đầu, KHÔNG phải ngày Airflow chạy thật
BATCH_PER_DAY = 4  # 100 dòng test / 4 mỗi ngày ~ 25 ngày mô phỏng


def task_assign_sim_dates(**context):
    """Chạy 1 lần, idempotent: gán sim_date cho từng dòng test set nếu chưa có."""
    path = f"{DATA_DIR}/splits/test_with_dates.csv"
    df_test = pd.read_csv(f"{DATA_DIR}/splits/test.csv")
    df_test = df_test.sort_values("id").reset_index(drop=True)
    df_test["sim_date"] = [
        (SIM_START_DATE + timedelta(days=i // BATCH_PER_DAY)).strftime("%Y-%m-%d")
        for i in range(len(df_test))
    ]
    df_test.to_csv(path, index=False)
    print(f"Đã gán sim_date cho {len(df_test)} dòng, từ {df_test['sim_date'].min()} đến {df_test['sim_date'].max()}")


def task_predict_daily_batch(**context):
    sys.path.insert(0, "/opt/airflow/project")
    from shared.inference import load_model_bundle, predict_from_silver
    from shared.db import init_tables, write_predictions

    # logical_date của Airflow -> map sang sim_date để demo dễ (dùng ds trực tiếp làm sim_date)
    ds = context["ds"]  # 'YYYY-MM-DD' theo logical_date của DAG run
    df_test = pd.read_csv(f"{DATA_DIR}/splits/test_with_dates.csv")

    # Demo mapping: coi ds của DAG run như sim_date luôn cho đơn giản
    batch = df_test[df_test["sim_date"] == ds]
    if batch.empty:
        print(f"Không có batch nào cho ngày giả lập {ds} — bỏ qua (đủ hết dữ liệu test).")
        return

    bundle = load_model_bundle(MODELS_DIR)
    preds_raw, preds_clean = predict_from_silver(batch, bundle)

    out = batch[["id", "customer_id"]].copy()
    out["sim_date"] = ds
    out["predicted_nps_raw"] = preds_raw
    out["predicted_nps"] = preds_clean
    out["actual_nps"] = batch[TARGET_COL].values

    init_tables()
    write_predictions(out, source="daily_batch", model_name=bundle["meta"]["model_name"])
    print(f"Đã predict + ghi {len(out)} dòng cho ngày {ds}")


default_args = {"owner": "tung", "retries": 1}

with DAG(
    dag_id="dag_daily_predict",
    description="Giả lập batch data đến mỗi ngày (từ test set) -> predict -> ghi Postgres",
    default_args=default_args,
    schedule="@daily",
    start_date=SIM_START_DATE,
    end_date=SIM_START_DATE + timedelta(days=30),  # đủ để phủ hết ~25 ngày mô phỏng
    catchup=True,  # chạy bù hết các ngày quá khứ -> mô phỏng nhanh toàn bộ lịch sử
    max_active_runs=1,
    tags=["nps", "predict", "daily"],
) as dag:

    assign_dates_task = PythonOperator(
        task_id="assign_sim_dates", python_callable=task_assign_sim_dates
    )
    predict_task = PythonOperator(
        task_id="predict_daily_batch", python_callable=task_predict_daily_batch
    )

    assign_dates_task >> predict_task
