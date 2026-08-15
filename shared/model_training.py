"""
Gộp logic train + so sánh model vào MỘT nơi duy nhất, dùng chung cho:
- scripts/train.py (chạy CLI, dev/test nhanh không cần Airflow)
- airflow/dags/dag_train.py (chạy trong Airflow)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from .preprocessing import add_features


def compare_models(X_train_raw: pd.DataFrame, y_train: pd.Series,
                    X_dev_raw: pd.DataFrame, y_dev: pd.Series) -> dict:
    """
    Train + so sánh 7 model (Ridge, DecisionTree, RandomForest, GradientBoosting,
    XGBoost, MLP, Stacking) trên train, chọn theo dev MAE thấp nhất.

    Trả về dict:
        {
          "results_sorted": [(name, params, model, dev_mae, scale_mode), ...],  # sort theo dev_mae tăng dần
          "winner": (name, params, model, dev_mae, scale_mode),
          "scaler": StandardScaler đã fit trên X_train (dùng cho model cần scale),
          "feature_columns": list cột sau add_features (thứ tự chuẩn để predict sau này),
        }
    """
    X_train = add_features(X_train_raw)
    X_dev = add_features(X_dev_raw)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_dev_s = scaler.transform(X_dev)

    results = []

    # 1. Ridge
    best = None
    for alpha in [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]:
        m = Ridge(alpha=alpha).fit(X_train_s, y_train)
        mae_d = mean_absolute_error(y_dev, m.predict(X_dev_s))
        if best is None or mae_d < best[3]:
            best = ("Ridge", {"alpha": alpha}, m, mae_d, "scaled")
    results.append(best)

    # 2. Decision Tree
    best = None
    for depth in [3, 5, 7, 10, None]:
        m = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(X_train, y_train)
        mae_d = mean_absolute_error(y_dev, m.predict(X_dev))
        if best is None or mae_d < best[3]:
            best = ("DecisionTree", {"max_depth": depth}, m, mae_d, "raw")
    results.append(best)

    # 3. Random Forest
    best = None
    for n_est in [200, 400, 600]:
        for depth in [5, 10, 15, None]:
            for min_leaf in [1, 2]:
                m = RandomForestRegressor(
                    n_estimators=n_est, max_depth=depth,
                    min_samples_leaf=min_leaf, random_state=42, n_jobs=-1
                ).fit(X_train, y_train)
                mae_d = mean_absolute_error(y_dev, m.predict(X_dev))
                if best is None or mae_d < best[3]:
                    best = ("RandomForest",
                            {"n_estimators": n_est, "max_depth": depth, "min_samples_leaf": min_leaf},
                            m, mae_d, "raw")
    results.append(best)

    # 4. Gradient Boosting
    best = None
    for n_est in [200, 400, 600]:
        for depth in [3, 4, 5]:
            for lr in [0.03, 0.05, 0.1, 0.15]:
                for subsample in [0.7, 0.8, 1.0]:
                    m = GradientBoostingRegressor(
                        n_estimators=n_est, max_depth=depth, learning_rate=lr,
                        subsample=subsample, min_samples_leaf=3, random_state=42
                    ).fit(X_train, y_train)
                    mae_d = mean_absolute_error(y_dev, m.predict(X_dev))
                    if best is None or mae_d < best[3]:
                        best = ("GradientBoosting",
                                {"n_estimators": n_est, "max_depth": depth,
                                 "learning_rate": lr, "subsample": subsample},
                                m, mae_d, "raw")
    results.append(best)

    # 5. XGBoost (early stopping)
    best = None
    for depth in [3, 4, 5, 6]:
        for lr in [0.02, 0.05, 0.1]:
            for subsample in [0.7, 0.8, 1.0]:
                for col_sub in [0.7, 0.8, 1.0]:
                    for min_cw in [1, 3, 5]:
                        m = XGBRegressor(
                            n_estimators=2000,
                            max_depth=depth, learning_rate=lr,
                            subsample=subsample, colsample_bytree=col_sub,
                            min_child_weight=min_cw,
                            reg_alpha=0.1, reg_lambda=1.0,
                            random_state=42, verbosity=0, n_jobs=-1,
                            early_stopping_rounds=50,
                        ).fit(
                            X_train, y_train,
                            eval_set=[(X_dev, y_dev)],
                            verbose=False,
                        )
                        mae_d = mean_absolute_error(y_dev, m.predict(X_dev))
                        if best is None or mae_d < best[3]:
                            best = ("XGBoost",
                                    {"n_estimators": m.best_iteration, "max_depth": depth,
                                     "learning_rate": lr, "subsample": subsample,
                                     "colsample_bytree": col_sub, "min_child_weight": min_cw},
                                    m, mae_d, "raw")
    results.append(best)

    # 6. MLP
    best = None
    for hidden in [(128, 64), (64, 64, 32), (256, 128, 64), (128, 64, 32)]:
        for lr_init in [0.001, 0.0005, 0.0001]:
            for alpha in [0.0001, 0.001]:
                m = MLPRegressor(
                    hidden_layer_sizes=hidden, max_iter=5000,
                    learning_rate_init=lr_init, alpha=alpha,
                    random_state=42,
                    early_stopping=True, validation_fraction=0.1, n_iter_no_change=40
                ).fit(X_train_s, y_train)
                mae_d = mean_absolute_error(y_dev, m.predict(X_dev_s))
                if best is None or mae_d < best[3]:
                    best = ("MLP", {"hidden_layer_sizes": hidden,
                                    "lr_init": lr_init, "alpha": alpha},
                            m, mae_d, "scaled")
    results.append(best)

    # 7. Stacking: RF + GB + XGB -> Ridge meta-learner
    rf_best = next(r for r in results if r[0] == "RandomForest")
    gb_best = next(r for r in results if r[0] == "GradientBoosting")
    xgb_best = next(r for r in results if r[0] == "XGBoost")

    stack_X_train = np.column_stack([
        rf_best[2].predict(X_train), gb_best[2].predict(X_train), xgb_best[2].predict(X_train),
    ])
    stack_X_dev = np.column_stack([
        rf_best[2].predict(X_dev), gb_best[2].predict(X_dev), xgb_best[2].predict(X_dev),
    ])
    meta_learner = Ridge(alpha=1.0).fit(stack_X_train, y_train)
    mae_stack = mean_absolute_error(y_dev, meta_learner.predict(stack_X_dev))
    results.append(("Stacking(RF+GB+XGB)", {}, meta_learner, mae_stack, "stacking"))

    results_sorted = sorted(results, key=lambda r: r[3])

    return {
        "results_sorted": results_sorted,
        "winner": results_sorted[0],
        "scaler": scaler,
        "feature_columns": list(X_train.columns),
    }


def retrain_winner_on_train_dev(winner_name: str, winner_params: dict,
                                 X_train_raw: pd.DataFrame, y_train: pd.Series,
                                 X_dev_raw: pd.DataFrame, y_dev: pd.Series):
    """
    Retrain model thắng trên train+dev gộp lại (nhiều data hơn -> test MAE thường tốt hơn).
    Bỏ qua nếu winner là Stacking (ensemble của các model khác, không retrain đơn giản được).
    Trả về (model, scaler) đã fit trên toàn bộ train+dev.
    """
    X_all_raw = pd.concat([X_train_raw, X_dev_raw], ignore_index=True)
    y_all = pd.concat([y_train, y_dev], ignore_index=True)
    X_all = add_features(X_all_raw)
    scaler = StandardScaler().fit(X_all)
    X_all_s = scaler.transform(X_all)

    if winner_name == "Ridge":
        model = Ridge(alpha=winner_params["alpha"]).fit(X_all_s, y_all)
    elif winner_name == "XGBoost":
        model = XGBRegressor(
            n_estimators=winner_params.get("n_estimators", 300),
            max_depth=winner_params["max_depth"],
            learning_rate=winner_params["learning_rate"],
            subsample=winner_params["subsample"],
            colsample_bytree=winner_params["colsample_bytree"],
            min_child_weight=winner_params.get("min_child_weight", 3),
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0, n_jobs=-1
        ).fit(X_all, y_all)
    elif winner_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=winner_params["n_estimators"],
            max_depth=winner_params["max_depth"],
            min_samples_leaf=winner_params["min_samples_leaf"],
            random_state=42, n_jobs=-1
        ).fit(X_all, y_all)
    elif winner_name == "GradientBoosting":
        model = GradientBoostingRegressor(
            n_estimators=winner_params["n_estimators"],
            max_depth=winner_params["max_depth"],
            learning_rate=winner_params["learning_rate"],
            subsample=winner_params["subsample"],
            min_samples_leaf=3, random_state=42
        ).fit(X_all, y_all)
    elif winner_name == "MLP":
        model = MLPRegressor(
            hidden_layer_sizes=winner_params["hidden_layer_sizes"],
            max_iter=5000,
            learning_rate_init=winner_params["lr_init"],
            alpha=winner_params["alpha"],
            random_state=42,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=40
        ).fit(X_all_s, y_all)
    elif winner_name == "DecisionTree":
        model = DecisionTreeRegressor(
            max_depth=winner_params["max_depth"], random_state=42
        ).fit(X_all, y_all)
    else:
        return None, scaler  # Stacking: không retrain

    return model, scaler
