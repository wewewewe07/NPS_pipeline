"""
Chia dữ liệu Silver thành train/dev/test theo tỷ lệ 80:10:10.

Chiến lược stratify: Target=0 chỉ có 1 dòng duy nhất trong toàn bộ dataset,
không đủ mẫu để stratify trực tiếp theo từng giá trị Target (0-10).
=> Stratify theo nhóm NPS chuẩn: Detractor (0-6) / Passive (7-8) / Promoter (9-10),
vừa đảm bảo 3 tập có phân phối tương đồng, vừa đúng ý nghĩa nghiệp vụ NPS.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split
from shared.preprocessing import bronze_to_silver

RANDOM_STATE = 42


def nps_group(target: int) -> str:
    if target <= 6:
        return "detractor"
    elif target <= 8:
        return "passive"
    else:
        return "promoter"


def main():
    df_raw = pd.read_csv("data/bronze/nps_data2_id.csv")
    df_silver = bronze_to_silver(df_raw)
    df_silver.to_csv("data/silver/nps_silver.csv", index=False)

    strata = df_silver["target"].apply(nps_group)

    # Bước 1: tách 80% train, 20% temp (sẽ chia tiếp thành dev/test)
    df_train, df_temp, strata_train, strata_temp = train_test_split(
        df_silver, strata, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
    )
    # Bước 2: chia 20% temp thành 10% dev + 10% test (chia đôi)
    df_dev, df_test = train_test_split(
        df_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=strata_temp
    )

    df_train.to_csv("data/splits/train.csv", index=False)
    df_dev.to_csv("data/splits/dev.csv", index=False)
    df_test.to_csv("data/splits/test.csv", index=False)

    print(f"Train: {len(df_train)} dòng ({len(df_train)/len(df_silver):.1%})")
    print(f"Dev:   {len(df_dev)} dòng ({len(df_dev)/len(df_silver):.1%})")
    print(f"Test:  {len(df_test)} dòng ({len(df_test)/len(df_silver):.1%})")

    print("\nPhân phối nhóm NPS theo tập:")
    for name, d in [("train", df_train), ("dev", df_dev), ("test", df_test)]:
        dist = d["target"].apply(nps_group).value_counts(normalize=True).round(3)
        print(f"  {name}: {dist.to_dict()}")


if __name__ == "__main__":
    main()
