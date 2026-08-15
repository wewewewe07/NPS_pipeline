"""
Streamlit dashboard - 3 tab:
1. Data Analysis: phân phối feature gốc
2. Model Performance: MAE/drift theo sim_date (đọc từ bảng predictions do dag_daily_predict ghi)
3. Predict: user input data -> gọi FastAPI /predic`t -> hiển thị + tải kết quả
"""
import os
import sys
sys.path.insert(0, "/app")

import requests
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine

APP_DB_URL = os.environ.get("APP_DB_URL", "postgresql+psycopg2://airflow:airflow@localhost:5432/nps_app")
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="NPS Pipeline Dashboard", layout="wide")
st.title("🏨 NPS Prediction Pipeline — Dashboard")

engine = create_engine(APP_DB_URL)

tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "📈 Model Performance", "📤 Upload & Predict"])

# ---------------- TAB 1: DATA ANALYSIS ----------------
with tab1:
    st.subheader("Phân phối dữ liệu gốc")
    try:
        df = pd.read_csv("/app/data/silver/nps_silver.csv")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Target (NPS) distribution**")
            st.bar_chart(df["target"].value_counts().sort_index())
            st.write("**Feedback rating**")
            st.bar_chart(df["feedback_rating"].value_counts().sort_index())
        with col2:
            st.write("**Sentiment score**")
            st.bar_chart(df["sentiment_score"].value_counts().sort_index())
            st.write("**Age distribution**")
            st.bar_chart(df["age"].value_counts(bins=10).sort_index())

        st.write("**Tương quan với Target (top features)**")
        corr = pd.read_csv("/app/data/gold/correlation_with_target.csv", index_col=0)
        st.bar_chart(corr["correlation"])
    except FileNotFoundError:
        st.warning("Chưa có data Silver/Gold — hãy chạy dag_etl trước.")

# ---------------- TAB 2: MODEL PERFORMANCE ----------------
with tab2:
    st.subheader("Kết quả predict theo ngày giả lập (dag_daily_predict)")
    try:
        pred_df = pd.read_sql(
            "SELECT * FROM predictions WHERE source = 'daily_batch' ORDER BY sim_date", engine
        )
        if pred_df.empty:
            st.info("Chưa có prediction nào — hãy chạy DAG dag_daily_predict trên Airflow.")
        else:
            pred_df["abs_error"] = (pred_df["predicted_nps"] - pred_df["actual_nps"]).abs()
            daily_mae = pred_df.groupby("sim_date")["abs_error"].mean()

            st.write("**MAE theo ngày giả lập**")
            st.line_chart(daily_mae)

            st.write("**Predicted vs Actual NPS (scatter theo ngày gần nhất)**")
            st.scatter_chart(pred_df, x="actual_nps", y="predicted_nps")

            st.write(f"**MAE tổng (toàn bộ predictions):** {pred_df['abs_error'].mean():.4f}")
            st.dataframe(pred_df.tail(20))
    except Exception as e:
        st.warning(f"Chưa kết nối được tới bảng predictions: {e}")

    st.subheader("Model registry")
    try:
        reg_df = pd.read_sql("SELECT * FROM model_registry ORDER BY trained_at DESC", engine)
        st.dataframe(reg_df)
    except Exception as e:
        st.info("Chưa có model nào trong registry — hãy chạy dag_train.")

# Mapping mã tỉnh/thành Nhật Bản (1-47) -> tên tiếng Nhật
PREFECTURE_MAP = {
    1: "Hokkaido", 2: "Aomori", 3: "Iwate", 4: "Miyagi", 5: "Akita",
    6: "Yamagata", 7: "Fukushima", 8: "Ibaraki", 9: "Tochigi", 10: "Gunma",
    11: "Saitama", 12: "Chiba", 13: "Tokyo", 14: "Kanagawa", 15: "Niigata",
    16: "Toyama", 17: "Ishikawa", 18: "Fukui", 19: "Yamanashi", 20: "Nagano",
    21: "Gifu", 22: "Shizuoka", 23: "Aichi", 24: "Mie", 25: "Shiga",
    26: "Kyoto", 27: "Osaka", 28: "Hyogo", 29: "Nara", 30: "Wakayama",
    31: "Tottori", 32: "Shimane", 33: "Okayama", 34: "Hiroshima", 35: "Yamaguchi",
    36: "Tokushima", 37: "Kagawa", 38: "Ehime", 39: "Kochi", 40: "Fukuoka",
    41: "Saga", 42: "Nagasaki", 43: "Kumamoto", 44: "Oita", 45: "Miyazaki",
    46: "Kagoshima", 47: "Okinawa",
}

# ---------------- TAB 3: PREDICT NPS ----------------
with tab3:
    st.subheader("Dự đoán NPS")
    upload_tab, manual_tab = st.tabs(["📂 Upload file CSV", "✏️ Nhập tay"])

    # ---- Upload CSV ----
    with upload_tab:
        st.caption("File cần đúng schema gốc (Id, CID, 年齢, 性別, 都道府県, ...)")
        uploaded = st.file_uploader("Chọn file CSV", type=["csv"])
        if uploaded is not None:
            preview_df = pd.read_csv(pd.io.common.BytesIO(uploaded.getvalue()))
            st.write(f"**Preview** ({len(preview_df)} dòng):")
            st.dataframe(preview_df.head(5), use_container_width=True)
            if st.button("🔮 Predict", key="btn_upload"):
                with st.spinner("Đang gọi API predict..."):
                    resp = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                    )
                if resp.status_code == 200:
                    result_df = pd.read_csv(pd.io.common.BytesIO(resp.content))
                    st.success(f"Đã predict {len(result_df)} dòng")
                    st.dataframe(result_df, use_container_width=True)
                    st.download_button(
                        "⬇️ Tải kết quả (CSV)", data=resp.content,
                        file_name="predictions.csv", mime="text/csv",
                    )
                else:
                    st.error(f"Lỗi API ({resp.status_code}): {resp.text}")

    # ---- Nhập tay ----
    with manual_tab:
        st.caption("Điền đầy đủ các trường bên dưới rồi nhấn **Dự đoán**")

        with st.form("manual_predict_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🧑 Thông tin khách hàng**")
                age = st.number_input(
                    "年齢 / Tuổi", min_value=15, max_value=100, value=30, step=1
                )
                gender = st.selectbox(
                    "性別 / Giới tính",
                    options=["M", "F"],
                    format_func=lambda x: "Nam (M)" if x == "M" else "Nữ (F)",
                )
                prefecture = st.selectbox(
                    "都道府県 / Tỉnh/thành phố",
                    options=list(PREFECTURE_MAP.keys()),
                    index=12,  # mặc định: Tokyo (13)
                    format_func=lambda code: f"{PREFECTURE_MAP[code]} ({code})",
                )
                feedback_rating = st.slider(
                    "評価フィードバック / Đánh giá phản hồi", min_value=1, max_value=5, value=3,
                    help="Điểm đánh giá từ 1 (rất tệ) đến 5 (rất tốt)"
                )
                sentiment_score = st.selectbox(
                    "感情分析結果 / Kết quả phân tích cảm xúc",
                    options=[-1, 0, 1],
                    format_func=lambda x: {-1: "Tệ", 0: "Bình thường", 1: "Tốt"}[x],
                    index=1,
                )

            with col2:
                st.markdown("**💰 Thông tin chi tiêu**")
                total_spend = st.number_input(
                    "購入総額 / Tổng chi tiêu (¥)", min_value=0, value=50000, step=1000
                )
                room_bookings = st.number_input(
                    "部屋予約回数 / Số lần đặt phòng", min_value=0, value=2, step=1
                )
                room_spend = st.number_input(
                    "部屋利用総額 / Chi tiêu phòng (¥)", min_value=0, value=40000, step=1000
                )
                restaurant_visits = st.number_input(
                    "レストラン利用回数 / Số lần dùng nhà hàng", min_value=0, value=3, step=1
                )
                restaurant_spend = st.number_input(
                    "レストラン利用総額 / Chi tiêu nhà hàng (¥)", min_value=0, value=10000, step=500
                )
                pool_uses = st.number_input(
                    "プール利用回数 / Số lần dùng hồ bơi", min_value=0, value=1, step=1
                )
                breakfast_uses = st.number_input(
                    "朝食利用回数 / Số lần dùng bữa sáng", min_value=0, value=2, step=1
                )

            submitted = st.form_submit_button("🔮 Dự đoán NPS", use_container_width=True)

        if submitted:
            payload = {
                "age": int(age),
                "gender": gender,
                "prefecture": int(prefecture),
                "feedback_rating": int(feedback_rating),
                "sentiment_score": int(sentiment_score),
                "total_spend": int(total_spend),
                "room_bookings": int(room_bookings),
                "room_spend": int(room_spend),
                "restaurant_visits": int(restaurant_visits),
                "restaurant_spend": int(restaurant_spend),
                "pool_uses": int(pool_uses),
                "breakfast_uses": int(breakfast_uses),
            }
            with st.spinner("Đang gọi API dự đoán..."):
                try:
                    resp = requests.post(f"{API_URL}/predict-single", json=payload)
                    if resp.status_code == 200:
                        result = resp.json()
                        nps = result["predicted_nps"]
                        nps_raw = result["predicted_nps_raw"]

                        if nps >= 9:
                            label, color = "Promoter 🌟", "green"
                        elif nps >= 7:
                            label, color = "Passive 😐", "orange"
                        else:
                            label, color = "Detractor ⚠️", "red"

                        st.success("Dự đoán thành công!")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("NPS dự đoán (đã làm tròn)", f"{nps} / 10")
                        c2.metric("NPS thô (raw)", f"{nps_raw}")
                        c3.metric("Phân loại khách hàng", label)

                        st.info(
                            f"**Model:** {result.get('model_name', 'N/A')}  |  "
                            f"**Dev MAE:** {result.get('dev_mae', 'N/A')}"
                        )
                    else:
                        st.error(f"Lỗi API ({resp.status_code}): {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Không kết nối được tới API. Vui lòng kiểm tra service đang chạy.")
