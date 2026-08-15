"""
Train + so sánh model, chọn best dựa trên dev MAE.
Toàn bộ logic so sánh model nằm ở shared/model_training.py (dùng chung với dag_train.py)
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import joblib
from shared.model_training import compare_models, retrain_winner_on_train_dev

TARGET_COL = "target"


def load_gold(split):
    df = pd.read_csv(f"data/gold/{split}_gold.csv")
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


def main():
    X_train_raw, y_train = load_gold("train")
    X_dev_raw, y_dev = load_gold("dev")

    result = compare_models(X_train_raw, y_train, X_dev_raw, y_dev)

    print(f"\n{'Model':<30} {'Best params':<60} {'Dev MAE':>10}")
    print("-" * 102)
    for name, params, _, mae, _ in result["results_sorted"]:
        print(f"{name:<30} {str(params):<60} {mae:>10.4f}")

    winner_name, winner_params, winner_model, winner_mae, winner_scale = result["winner"]
    print(f"\n>>> Model tốt nhất trên dev: {winner_name} (dev MAE={winner_mae:.4f})")
    print(f"    Params: {winner_params}")

    scaler = result["scaler"]
    if winner_scale != "stacking":
        print("\n[Retrain] Re-training winner trên train+dev combined...")
        retrained_model, retrained_scaler = retrain_winner_on_train_dev(
            winner_name, winner_params, X_train_raw, y_train, X_dev_raw, y_dev
        )
        winner_model = retrained_model
        scaler = retrained_scaler

    joblib.dump(winner_model, "models/best_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    with open("models/model_metadata.json", "w") as f:
        json.dump({
            "model_name": winner_name,
            "params": winner_params,
            "dev_mae": winner_mae,
            "needs_scaling": winner_scale == "scaled",
            "feature_columns": result["feature_columns"],
            "uses_feature_engineering": True,
        }, f, indent=2)

    print("\nĐã lưu: models/best_model.joblib, models/model_metadata.json, models/scaler.joblib")


if __name__ == "__main__":
    main()
