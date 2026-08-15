# NPS Prediction Pipeline — Hotel Japan Dataset

Portfolio project: end-to-end pipeline dự đoán NPS (Net Promoter Score) cho khách sạn,
từ ETL (Bronze/Silver/Gold) → train/select model → daily batch predict (Airflow) →
serving API (FastAPI) → dashboard (Streamlit).

## Kiến trúc

```
Bronze (raw CSV) -> Silver (cleaned) -> Gold (encoded, ready for ML)
                                            |
                          +-----------------+------------------+
                          |                                    |
                    Airflow dag_train                   Airflow dag_daily_predict
                 (so sánh 5 model, chọn best)          (giả lập data đến mỗi ngày,
                          |                              predict, ghi Postgres)
                          v                                    |
                  models/best_model.joblib                     v
                          |                              predictions table
                          +--------> FastAPI /predict <--------+
                                     (user upload file)         |
                                            |                   v
                                            +----------> Streamlit Dashboard
```

## Cấu trúc thư mục

```
nps-pipeline/
├── docker-compose.yml
├── postgres-init/init-multiple-db.sh   # tạo thêm db nps_app cạnh db airflow
├── airflow/dags/
│   ├── dag_etl.py             # Bronze -> Silver -> Gold + split 80/10/10
│   ├── dag_train.py           # train + so sánh 5 model, chọn best, ghi registry
│   └── dag_daily_predict.py   # giả lập batch data theo ngày, predict, ghi Postgres
├── shared/
│   ├── preprocessing.py       # bronze_to_silver, silver_to_gold, add_features
│   ├── inference.py           # load_model_bundle, predict_from_silver (dùng chung mọi nơi)
│   ├── model_training.py      # train + so sánh model (dùng chung CLI + Airflow)
│   └── db.py                  # helper Postgres (predictions, model_registry)
├── scripts/                   # bản CLI độc lập (chạy không cần Airflow), dùng để dev/test
│   ├── split_data.py          
│   ├── correlation_analysis.py
│   ├── train.py                
│   ├── evaluate.py             
│   └── predict.py              
├── api/                        # FastAPI serving (/predict endpoint)
├── dashboard/                  # Streamlit dashboard
├── data/                       # bronze/silver/gold/splits (mount volume)
└── models/                     # best_model.joblib, encoder.joblib, metadata (mount volume)
```

## Cách chạy (cần Docker + Docker Compose)

### 1. Chạy các script CLI trước để có model + data sẵn (nhanh, không cần Airflow)

```bash
cd nps-pipeline
pip install -r api/requirements.txt
python scripts/split_data.py
python scripts/build_gold.py
python scripts/correlation_analysis.py
python scripts/train.py
python scripts/evaluate.py
python scripts/predict.py --input data/sample_new_input.csv --output data/sample_predictions.csv
```

Sau bước này, `models/` đã có `best_model.joblib`, `encoder.joblib`, `model_metadata.json`
— các service Docker (API, Airflow) sẽ dùng lại đúng các file này.

### 2. Khởi động toàn bộ stack

```bash
docker compose up --build
```

Các service:
- **Airflow UI**: http://localhost:8080 (user: `admin` / pass: `admin`)
- **FastAPI**: http://localhost:8000/docs (Swagger UI để test `/predict`)
- **Streamlit dashboard**: http://localhost:8501
- **Postgres**: localhost:5432 (db `airflow` cho Airflow metadata, `nps_app` cho predictions/model_registry)

### 3. Trigger DAGs (lần đầu, theo thứ tự)

Trên Airflow UI (http://localhost:8080), bật và trigger thủ công theo thứ tự:
1. `dag_etl` — chạy 1 lần để tạo Silver/Gold/splits
2. `dag_train` — chạy 1 lần (hoặc mỗi khi muốn retrain) để chọn model + ghi registry
3. `dag_daily_predict` — `catchup=True` đã được cấu hình sẵn, Airflow sẽ tự chạy bù
   toàn bộ ~25 "ngày giả lập" (xem `SIM_START_DATE`/`BATCH_PER_DAY` trong DAG để chỉnh)

Hoặc trigger qua CLI trong container:
```bash
docker compose exec airflow-webserver airflow dags trigger dag_etl
docker compose exec airflow-webserver airflow dags trigger dag_train
docker compose exec airflow-webserver airflow dags unpause dag_daily_predict
```

### 4. Xem kết quả trên Dashboard

Mở http://localhost:8501:
- Tab **Data Analysis**: phân phối feature, tương quan với Target
- Tab **Model Performance**: MAE theo ngày giả lập, predicted vs actual, model registry
- Tab **Predict NPS**: nhập tay thông tin khách hàng → nhận NPS dự đoán ngay (Promoter/Passive/Detractor)

## Ghi chú

- **Không có cột ngày thật** trong dataset gốc → `dag_daily_predict` giả lập bằng cách
  rải 100 dòng test set ra ~25 ngày (`SIM_START_DATE` trong `dag_daily_predict.py`).
- **Encoder & model chỉ fit 1 lần lúc train** (trên tập train), lưu lại và dùng chung cho
  cả `predict.py`, FastAPI, và `dag_daily_predict` — tránh training-serving skew.
- **Test set chỉ dùng đúng 1 lần** để tính MAE cuối cùng, không được dùng để tune
  hyperparameter (việc đó dùng dev set).
- Kết quả đã đạt được: **XGBoost**, Dev MAE 0.7104, **Test MAE 0.6887**.
