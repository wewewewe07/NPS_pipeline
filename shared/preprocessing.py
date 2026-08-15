"""
Shared preprocessing module dùng chung cho:
- ETL pipeline (Bronze -> Silver -> Gold)
- Training pipeline
- Predict-from-file script
- FastAPI serving
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Mapping tên cột tiếng Nhật -> tiếng Anh, cho code dễ đọc hơn
COLUMN_MAP = {
    "Id": "id",
    "CID": "customer_id",
    "年齢": "age",
    "性別": "gender",
    "都道府県": "prefecture",
    "評価フィードバック": "feedback_rating",
    "感情分析結果": "sentiment_score",
    "購入総額": "total_spend",
    "部屋予約回数": "room_bookings",
    "部屋利用総額": "room_spend",
    "レストラン利用回数": "restaurant_visits",
    "レストラン利用総額": "restaurant_spend",
    "プール利用回数": "pool_uses",
    "朝食利用回数": "breakfast_uses",
    "Target": "target",
}

# Các cột feature dùng để train (từ cột 3 đến cột gần cuối)
# bỏ id, customer_id (cột 1,2) và target (cột cuối)
ID_COLS = ["id", "customer_id"]
TARGET_COL = "target"
FEATURE_COLS_RAW = [
    "age", "gender", "prefecture", "feedback_rating", "sentiment_score",
    "total_spend", "room_bookings", "room_spend",
    "restaurant_visits", "restaurant_spend", "pool_uses", "breakfast_uses",
]
CATEGORICAL_COLS = ["gender", "prefecture"]
NUMERIC_COLS = [c for c in FEATURE_COLS_RAW if c not in CATEGORICAL_COLS]


def bronze_to_silver(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Bronze -> Silver: rename cột, ép kiểu, xử lý data quality issue đã phát hiện.
    """
    df = df_raw.rename(columns=COLUMN_MAP).copy()

    # Data quality issue đã phát hiện: 2 cặp CID trùng nhưng thuộc tính khác nhau
    # (rất có thể là ID sinh ngẫu nhiên bị đụng, không phải cùng 1 khách hàng thật).
    # Xử lý: giữ nguyên id (unique) làm khoá chính, không dùng customer_id để join/group.
    dup_cid = df["customer_id"].duplicated(keep=False)
    if dup_cid.any():
        df.loc[dup_cid, "customer_id"] = (
            df.loc[dup_cid, "customer_id"] + "_" + df.loc[dup_cid, "id"].astype(str)
        )

    # Ép kiểu categorical
    df["gender"] = df["gender"].astype("category")
    df["prefecture"] = df["prefecture"].astype("category")

    # Validate: total_spend phải bằng room_spend + restaurant_spend
    calc_total = df["room_spend"] + df["restaurant_spend"]
    mismatch = (calc_total != df["total_spend"]).sum()
    if mismatch > 0:
        print(f"[WARN] {mismatch} dòng có total_spend không khớp room+restaurant spend")

    return df


def silver_to_gold(df_silver: pd.DataFrame, encoder=None, fit_encoder: bool = False):
    """
    Silver -> Gold: encode categorical -> feature matrix sẵn sàng cho sklearn/xgboost.
    - fit_encoder=True: dùng khi train, sẽ fit OneHotEncoder mới.
    - fit_encoder=False: dùng khi predict, PHẢI truyền encoder đã fit từ lúc train.
    Trả về: (X: DataFrame, encoder)
    """
    from sklearn.preprocessing import OneHotEncoder

    df = df_silver.copy()

    if fit_encoder:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(df[CATEGORICAL_COLS])
    if encoder is None:
        raise ValueError("Phải truyền encoder đã fit khi fit_encoder=False")

    cat_encoded = encoder.transform(df[CATEGORICAL_COLS])
    cat_cols_out = encoder.get_feature_names_out(CATEGORICAL_COLS)
    df_cat = pd.DataFrame(cat_encoded, columns=cat_cols_out, index=df.index)

    X = pd.concat([df[NUMERIC_COLS], df_cat], axis=1)
    return X, encoder


def save_encoder(encoder, path: str):
    joblib.dump(encoder, path)


def load_encoder(path: str):
    return joblib.load(path)


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering bổ sung sau silver_to_gold, dùng chung cho train và predict
    để đảm bảo training-serving parity.
    """
    df = X.copy()

    total = df["total_spend"].replace(0, np.nan)
    df["restaurant_ratio"] = df["restaurant_spend"] / total
    df["room_ratio"] = df["room_spend"] / total

    df["services_used"] = (
        (df["restaurant_visits"] > 0).astype(int)
        + (df["pool_uses"] > 0).astype(int)
        + (df["breakfast_uses"] > 0).astype(int)
    )

    bookings = df["room_bookings"].replace(0, np.nan)
    df["spend_per_booking"] = df["total_spend"] / bookings

    df["log_total_spend"] = np.log1p(df["total_spend"])
    df["log_room_spend"] = np.log1p(df["room_spend"])
    df["log_restaurant_spend"] = np.log1p(df["restaurant_spend"])

    if "feedback_rating" in df.columns and "sentiment_score" in df.columns:
        df["feedback_x_sentiment"] = df["feedback_rating"] * df["sentiment_score"]

    total_visits = df["restaurant_visits"] + df["pool_uses"] + df["breakfast_uses"]
    df["visit_intensity"] = total_visits / bookings

    rest_visits = df["restaurant_visits"].replace(0, np.nan)
    df["restaurant_per_visit"] = df["restaurant_spend"] / rest_visits

    df["high_spender"] = (df["total_spend"] > df["total_spend"].median()).astype(int)
    df["loyal_customer"] = (df["room_bookings"] > df["room_bookings"].median()).astype(int)

    df = df.fillna(df.median(numeric_only=True))
    return df
