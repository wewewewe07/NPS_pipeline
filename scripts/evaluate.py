"""
Đánh giá MAE của model đã chọn trên TEST set (dùng đúng 1 lần).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from shared.inference import load_model_bundle, predict_from_gold

TARGET_COL = "target"


def main():
    bundle = load_model_bundle("models")
    meta = bundle["meta"]

    df_test = pd.read_csv("data/gold/test_gold.csv")
    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL]

    y_pred, _ = predict_from_gold(X_test, bundle)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"Model: {meta['model_name']} (params={meta['params']})")
    print(f"Dev MAE (lúc chọn model):  {meta['dev_mae']:.4f}")
    print(f"Test MAE (đánh giá cuối):  {mae:.4f}")
    print(f"Test RMSE:                 {rmse:.4f}")
    print(f"Test R^2:                  {r2:.4f}")

    meta["test_mae"] = mae
    meta["test_rmse"] = rmse
    meta["test_r2"] = r2
    with open("models/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
