"""
Phân tích tương quan các cột (trừ cột 1,2 - id, customer_id) với Target.
Tính trên tập TRAIN (không đụng dev/test để tránh nhìn trước dữ liệu).
"""
import pandas as pd

def main():
    df = pd.read_csv("data/gold/train_gold.csv")
    corr = df.corr(numeric_only=True)["target"].drop("target").sort_values(key=abs, ascending=False)
    corr.to_csv("data/gold/correlation_with_target.csv", header=["correlation"])
    print("Tương quan Pearson với Target (tập train, sắp xếp theo |correlation| giảm dần):\n")
    print(corr.to_string())

if __name__ == "__main__":
    main()
