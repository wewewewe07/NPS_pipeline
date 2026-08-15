"""
FastAPI serving cho tính năng "user upload file -> output NPS dự đoán"
"""
import io
import sys
sys.path.insert(0, "/app")

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from shared.preprocessing import bronze_to_silver, ID_COLS
from shared.inference import load_model_bundle, predict_from_silver
from shared.db import init_tables, write_predictions

app = FastAPI(title="NPS Predict API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "/app/models"
_bundle = None  # lazy-load, load 1 lần khi có request đầu tiên


def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = load_model_bundle(MODELS_DIR)
    return _bundle


class SingleRecord(BaseModel):
    """Schema cho dự đoán 1 record nhập tay từ dashboard."""
    age: int = Field(..., ge=15, le=100, description="年齢 / Age")
    gender: str = Field(..., description="性別 / Gender: M or F")
    prefecture: int = Field(..., description="都道府県 / Prefecture code")
    feedback_rating: int = Field(..., ge=1, le=5, description="評価フィードバック / Feedback rating 1-5")
    sentiment_score: int = Field(..., description="感情分析結果 / Sentiment: -1, 0, 1")
    total_spend: int = Field(..., ge=0, description="購入総額 / Total spend")
    room_bookings: int = Field(..., ge=0, description="部屋予約回数 / Room bookings")
    room_spend: int = Field(..., ge=0, description="部屋利用総額 / Room spend")
    restaurant_visits: int = Field(..., ge=0, description="レストラン利用回数 / Restaurant visits")
    restaurant_spend: int = Field(..., ge=0, description="レストラン利用総額 / Restaurant spend")
    pool_uses: int = Field(..., ge=0, description="プール利用回数 / Pool uses")
    breakfast_uses: int = Field(..., ge=0, description="朝食利用回数 / Breakfast uses")


@app.on_event("startup")
def startup():
    init_tables()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return get_bundle()["meta"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Chỉ hỗ trợ file .csv")

    content = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Không đọc được file CSV: {e}")

    try:
        df_silver = bronze_to_silver(df_raw)
        bundle = get_bundle()
        preds_raw, preds_clean = predict_from_silver(df_silver, bundle)
    except Exception as e:
        raise HTTPException(422, f"Lỗi xử lý dữ liệu / predict: {e}")

    id_cols_present = [c for c in ID_COLS if c in df_silver.columns]
    out = df_silver[id_cols_present].copy()
    out["predicted_nps_raw"] = preds_raw.round(3)
    out["predicted_nps"] = preds_clean

    try:
        log_df = out[["id", "customer_id", "predicted_nps", "predicted_nps_raw"]].copy()
        log_df["sim_date"] = None
        log_df["actual_nps"] = None
        write_predictions(log_df, source="user_upload", model_name=bundle["meta"]["model_name"])
    except Exception as e:
        print(f"[WARN] Không ghi được log vào Postgres: {e}")

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.post("/predict-single")
def predict_single(record: SingleRecord):
    """Dự đoán NPS cho 1 record nhập tay từ dashboard."""
    row = {
        "Id": 0,
        "CID": "manual_input",
        "年齢": record.age,
        "性別": record.gender,
        "都道府県": record.prefecture,
        "評価フィードバック": record.feedback_rating,
        "感情分析結果": record.sentiment_score,
        "購入総額": record.total_spend,
        "部屋予約回数": record.room_bookings,
        "部屋利用総額": record.room_spend,
        "レストラン利用回数": record.restaurant_visits,
        "レストラン利用総額": record.restaurant_spend,
        "プール利用回数": record.pool_uses,
        "朝食利用回数": record.breakfast_uses,
    }
    df_raw = pd.DataFrame([row])

    try:
        df_silver = bronze_to_silver(df_raw)
        bundle = get_bundle()
        preds_raw, preds_clean = predict_from_silver(df_silver, bundle)
    except Exception as e:
        raise HTTPException(422, f"Lỗi predict: {e}")

    meta = bundle["meta"]
    return {
        "predicted_nps": int(preds_clean[0]),
        "predicted_nps_raw": round(float(preds_raw[0]), 3),
        "model_name": meta["model_name"],
        "dev_mae": meta.get("dev_mae"),
    }
