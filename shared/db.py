"""
Helper kết nối tới nps_app database (Postgres) - dùng chung cho:
- Airflow DAGs (ghi predictions, model registry)
- FastAPI (đọc/ghi predictions từ upload)
- Streamlit dashboard (đọc predictions để vẽ biểu đồ)
"""
import os
from sqlalchemy import create_engine, text

APP_DB_URL = os.environ.get(
    "APP_DB_URL", "postgresql+psycopg2://airflow:airflow@localhost:5432/nps_app"
)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(APP_DB_URL)
    return _engine


DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER,
    customer_id TEXT,
    sim_date DATE,
    predicted_nps INTEGER,
    predicted_nps_raw FLOAT,
    actual_nps INTEGER,
    model_name TEXT,
    source TEXT,          -- 'daily_batch' hoặc 'user_upload'
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name TEXT,
    params JSONB,
    dev_mae FLOAT,
    test_mae FLOAT,
    trained_at TIMESTAMP DEFAULT now(),
    is_active BOOLEAN DEFAULT true
);
"""


def init_tables():
    engine = get_engine()
    with engine.begin() as conn:
        for stmt in DDL.strip().split(";"):
            if stmt.strip():
                conn.execute(text(stmt))


def write_predictions(df, source: str, model_name: str):
    """df cần có: id, customer_id, sim_date (optional), predicted_nps, predicted_nps_raw, actual_nps (optional)"""
    engine = get_engine()
    df = df.copy()
    df["source"] = source
    df["model_name"] = model_name
    df.to_sql("predictions", engine, if_exists="append", index=False)


def write_model_registry(model_name, params, dev_mae, test_mae):
    import json
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("UPDATE model_registry SET is_active = false"))
        conn.execute(
            text(
                "INSERT INTO model_registry (model_name, params, dev_mae, test_mae) "
                "VALUES (:m, :p, :d, :t)"
            ),
            {"m": model_name, "p": json.dumps(params), "d": dev_mae, "t": test_mae},
        )
