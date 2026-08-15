"""
Build Gold layer (one-hot encode) từ train/dev/test Silver,
fit encoder trên TRAIN only, lưu encoder để dùng lại lúc predict.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from shared.preprocessing import silver_to_gold, save_encoder, FEATURE_COLS_RAW, TARGET_COL


def main():
    df_train = pd.read_csv("data/splits/train.csv")
    df_dev = pd.read_csv("data/splits/dev.csv")
    df_test = pd.read_csv("data/splits/test.csv")

    # Fit encoder CHỈ trên train -> tránh data leakage từ dev/test vào lúc train
    X_train, encoder = silver_to_gold(df_train, fit_encoder=True)
    X_dev, _ = silver_to_gold(df_dev, encoder=encoder, fit_encoder=False)
    X_test, _ = silver_to_gold(df_test, encoder=encoder, fit_encoder=False)

    save_encoder(encoder, "models/encoder.joblib")

    for name, X, df in [("train", X_train, df_train), ("dev", X_dev, df_dev), ("test", X_test, df_test)]:
        gold = X.copy()
        gold[TARGET_COL] = df[TARGET_COL].values
        gold.to_csv(f"data/gold/{name}_gold.csv", index=False)
        print(f"{name}_gold.csv: {gold.shape}")

    print(f"\nEncoder đã lưu: models/encoder.joblib")
    print(f"Feature columns sau encode: {list(X_train.columns)}")


if __name__ == "__main__":
    main()
