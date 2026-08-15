"""
Gộp toàn bộ logic "predict cho data mới" vào MỘT nơi duy nhất, để không còn
tình trạng mỗi nơi gọi model (evaluate.py, predict.py, dag_daily_predict.py, api)
tự viết lại pipeline preprocessing -> add_features -> scale -> predict và dễ quên bước.

Bất kỳ chỗ nào cần predict, chỉ cần:
    bundle = load_model_bundle()
    preds_raw, preds_clean = predict_from_silver(df_silver, bundle)
    # hoặc, nếu đã có sẵn Gold (đã one-hot encode):
    preds_raw, preds_clean = predict_from_gold(X_gold, bundle)
"""
import json
import joblib
import pandas as pd

from .preprocessing import silver_to_gold, load_encoder, add_features


def load_model_bundle(models_dir: str = "models") -> dict:
    """Load model + metadata + encoder + scaler (nếu cần) thành 1 bundle duy nhất."""
    with open(f"{models_dir}/model_metadata.json") as f:
        meta = json.load(f)

    model = joblib.load(f"{models_dir}/best_model.joblib")
    encoder = load_encoder(f"{models_dir}/encoder.joblib")
    scaler = joblib.load(f"{models_dir}/scaler.joblib") if meta.get("needs_scaling") else None

    return {"model": model, "meta": meta, "encoder": encoder, "scaler": scaler}


def _apply_feature_pipeline(X: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    Bước chung PHẢI áp dụng đúng thứ tự, giống hệt lúc train:
    (add_features nếu model được train với feature engineering) -> reindex đúng cột -> scale nếu cần.
    Đây chính là nơi bug "quên add_features" đã xảy ra 3 lần — giờ chỉ còn 1 chỗ để giữ đúng.
    """
    meta = bundle["meta"]
    if meta.get("uses_feature_engineering"):
        X = add_features(X)
    X = X[meta["feature_columns"]]
    if meta.get("needs_scaling"):
        X = bundle["scaler"].transform(X)
    return X


def predict_from_silver(df_silver: pd.DataFrame, bundle: dict):
    """
    Dùng khi input là dữ liệu Silver (chưa one-hot encode) — ví dụ: file user upload,
    batch từ dag_daily_predict, hoặc 1 record nhập tay.
    Trả về (preds_raw: np.array, preds_clean: np.array[int] đã clip 0-10 + round).
    """
    X, _ = silver_to_gold(df_silver, encoder=bundle["encoder"], fit_encoder=False)
    X = _apply_feature_pipeline(X, bundle)
    preds_raw = bundle["model"].predict(X)
    preds_clean = preds_raw.clip(0, 10).round().astype(int)
    return preds_raw, preds_clean


def predict_from_gold(X_gold: pd.DataFrame, bundle: dict):
    """
    Dùng khi input đã ở dạng Gold (đã one-hot encode, KHÔNG có cột target) —
    ví dụ: evaluate.py đọc thẳng từ data/gold/test_gold.csv.
    Trả về (preds_raw, preds_clean) giống predict_from_silver.
    """
    X = _apply_feature_pipeline(X_gold.copy(), bundle)
    preds_raw = bundle["model"].predict(X)
    preds_clean = preds_raw.clip(0, 10).round().astype(int)
    return preds_raw, preds_clean
