"""
Cho phép user nhập vào dữ liệu input (schema giống gốc, có thể
không có cột Target) và output ra NPS dự đoán cho từng dòng.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from shared.preprocessing import bronze_to_silver, ID_COLS
from shared.inference import load_model_bundle, predict_from_silver


def main():
    parser = argparse.ArgumentParser(description="Dự đoán NPS (Target) cho file input mới")
    parser.add_argument("--input", required=True, help="Đường dẫn file CSV input (schema giống dữ liệu gốc)")
    parser.add_argument("--output", required=True, help="Đường dẫn file CSV output (kèm cột predicted_nps)")
    args = parser.parse_args()

    df_raw = pd.read_csv(args.input)
    df_silver = bronze_to_silver(df_raw)

    bundle = load_model_bundle("models")
    preds_raw, preds_clean = predict_from_silver(df_silver, bundle)

    id_cols_present = [c for c in ID_COLS if c in df_silver.columns]
    out = df_silver[id_cols_present].copy()
    out["predicted_nps_raw"] = preds_raw.round(3)
    out["predicted_nps"] = preds_clean

    out.to_csv(args.output, index=False)
    print(f"Đã predict {len(out)} dòng -> lưu tại: {args.output}")
    print(out.head())


if __name__ == "__main__":
    main()
