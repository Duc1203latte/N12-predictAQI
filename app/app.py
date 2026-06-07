import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.cleaning  import clean_data
from src.feature  import build_features

# =============================================
# CẤU HÌNH TRANG
# =============================================
st.set_page_config(
    page_title = "PM2.5 Forecast Evaluator",
    page_icon  = "🌫️",
    layout     = "wide"
)

HORIZONS     = ['t+1', 't+2', 't+3']
TARGET_COLS  = ['pm25_t1', 'pm25_t2', 'pm25_t3']
MODEL_COLORS = {'Lasso': '#2E75B6', 'Ridge': '#ED7D31', 'RF': '#70AD47'}

# =============================================
# LOAD 3 MODEL (chỉ load 1 lần nhờ cache)
# =============================================
@st.cache_resource
def load_models():
    lasso = joblib.load('models/lasso_model.pkl')
    lasso_scaler = joblib.load('models/lasso_scaler.pkl')
    ridge = joblib.load('models/ridge_model.pkl')
    ridge_scaler = joblib.load('models/ridge_scaler.pkl')
    rf    = joblib.load('models/random_forest_model.pkl')
    return {
        'Lasso': (lasso, lasso_scaler),
        'Ridge': (ridge, ridge_scaler),
        'RF'   : (rf,    None),          # RF không cần scaler
    }

# =============================================
# HÀM TÍNH METRICS
# =============================================
def calc_metrics(y_true, y_pred):
    rows = []
    for i, h in enumerate(HORIZONS):
        err  = y_true[:, i] - y_pred[:, i]
        rmse = np.sqrt(np.mean(err**2))
        mae  = np.mean(np.abs(err))
        ss_res = np.sum(err**2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i]))**2)
        r2 = 1 - ss_res / ss_tot
        rows.append({'Horizon': h, 'RMSE': round(rmse, 2),
                     'MAE': round(mae, 2), 'R²': round(r2, 4)})
    return pd.DataFrame(rows)

# =============================================
# HÀM VẼ ACTUAL VS PREDICTED
# =============================================
def plot_actual_vs_pred(y_true, predictions, horizon_idx=0, n=300):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f'Actual vs Predicted — {HORIZONS[horizon_idx]}',
                 fontsize=13, fontweight='bold')

    for ax, (model_name, y_pred) in zip(axes, predictions.items()):
        color = MODEL_COLORS[model_name]
        ax.plot(range(n), y_true[:n, horizon_idx],
                color='#404040', linewidth=1.2, label='Actual', zorder=3)
        ax.plot(range(n), y_pred[:n, horizon_idx],
                color=color, linewidth=1.0, label=f'{model_name}', alpha=0.85)
        ax.fill_between(range(n),
                        y_true[:n, horizon_idx],
                        y_pred[:n, horizon_idx],
                        alpha=0.1, color=color)
        mae = np.mean(np.abs(y_true[:, horizon_idx] - y_pred[:, horizon_idx]))
        r2  = 1 - np.sum((y_true[:, horizon_idx] - y_pred[:, horizon_idx])**2) / \
                  np.sum((y_true[:, horizon_idx] - np.mean(y_true[:, horizon_idx]))**2)
        ax.set_title(f'{model_name}  —  MAE={mae:.2f} µg/m³  R²={r2:.4f}', fontsize=11)
        ax.set_ylabel('PM2.5 (µg/m³)', fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Giờ (tập Test)', fontsize=10)
    plt.tight_layout()
    return fig

# =============================================
# HÀM VẼ SCATTER
# =============================================
def plot_scatter(y_true, predictions, horizon_idx=0):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f'Scatter: Actual vs Predicted — {HORIZONS[horizon_idx]}',
                 fontsize=12, fontweight='bold')

    for ax, (model_name, y_pred) in zip(axes, predictions.items()):
        color = MODEL_COLORS[model_name]
        ax.scatter(y_true[:, horizon_idx], y_pred[:, horizon_idx],
                   color=color, alpha=0.2, s=5)
        lim = max(y_true[:, horizon_idx].max(), y_pred[:, horizon_idx].max())
        ax.plot([0, lim], [0, lim], 'k--', linewidth=1, label='Lý tưởng')
        r2 = 1 - np.sum((y_true[:, horizon_idx] - y_pred[:, horizon_idx])**2) / \
                 np.sum((y_true[:, horizon_idx] - np.mean(y_true[:, horizon_idx]))**2)
        ax.set_title(f'{model_name}  R²={r2:.4f}', fontsize=11)
        ax.set_xlabel('Actual', fontsize=9)
        ax.set_ylabel('Predicted', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig

# =============================================
# GIAO DIỆN CHÍNH
# =============================================
st.title("🌫️ PM2.5 Forecast — Model Evaluator")
st.markdown("Upload file CSV từ bất kỳ trạm nào để đánh giá và so sánh 3 model.")

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")
    uploaded_file = st.file_uploader(
        "Upload file CSV (Beijing Multisite)",
        type=["csv"],
        help="Ví dụ: PRSA_Data_Wanliu_20130301-20170228.csv"
    )
    split_ratio = st.slider("Tỉ lệ Train/Test", 0.6, 0.9, 0.8, 0.05,
                             help="Phần cuối dùng làm tập Test")
    horizon_choice = st.selectbox("Horizon hiển thị biểu đồ",
                                   ['t+1', 't+2', 't+3'], index=0)
    n_display = st.slider("Số giờ hiển thị trên biểu đồ", 100, 500, 300, 50)
    st.markdown("---")
    st.markdown("**Model đã train:** Dongsi 2013–2017")
    st.markdown("**Mục tiêu:** Dự báo PM2.5 tại t+1, t+2, t+3")

# ── Main ─────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Upload file CSV ở thanh bên trái để bắt đầu.")
    st.markdown("""
    **Các trạm có thể test:**
    `Aotizhongxin` · `Changping` · `Dingling` · `Dongsi` · `Guanyuan`
    `Gucheng` · `Huairou` · `Nongzhanguan` · `Shunyi` · `Tiantan`
    `Wanliu` · `Wanshouxigong`
    """)
    st.stop()

# ── Preprocessing ─────────────────────────────
with st.spinner("Đang xử lý dữ liệu..."):
    try:
        # Lưu file tạm để dùng clean_data
        tmp_path = f"data/raw/{uploaded_file.name}"
        os.makedirs("data/raw", exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        df_clean   = clean_data(tmp_path)
        df_feat    = build_features(df_clean)

        station_name = uploaded_file.name.replace("PRSA_Data_", "").split("_")[0]
        st.success(f"✅ Xử lý xong — Trạm: **{station_name}** | "
                   f"{df_feat.shape[0]:,} dòng | {df_feat.shape[1]} cột")

    except Exception as e:
        st.error(f"❌ Lỗi preprocessing: {e}")
        st.stop()

# ── Tách X, Y và scale ────────────────────────
X = df_feat.drop(columns=['datetime', 'PM2.5'] + TARGET_COLS)
Y = df_feat[TARGET_COLS]

split_idx  = int(len(df_feat) * split_ratio)
X_test     = X.iloc[split_idx:]
Y_test     = Y.iloc[split_idx:]
Y_true     = Y_test.values

models = load_models()

# Predict với 3 model
predictions = {}
for model_name, (model, scaler) in models.items():
    if scaler is not None:
        X_scaled = scaler.transform(X_test)    # dùng scaler đã fit trên Dongsi
    else:
        X_scaled = X_test                       # RF không cần scale
    predictions[model_name] = model.predict(X_scaled)

horizon_idx = HORIZONS.index(horizon_choice)

# ── Tabs ──────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Metrics", "📈 Actual vs Predicted", "⚪ Scatter"])

# ── Tab 1: Metrics ────────────────────────────
with tab1:
    st.subheader(f"So sánh 3 model — Trạm {station_name}")

    cols = st.columns(3)
    for col, (model_name, y_pred) in zip(cols, predictions.items()):
        df_m = calc_metrics(Y_true, y_pred)
        with col:
            color = MODEL_COLORS[model_name]
            # Dùng HTML color trực tiếp
            st.markdown(f"<h3 style='color:{color}'>{model_name}</h3>", unsafe_allow_html=True)
            st.dataframe(
                df_m.style
                    .highlight_min(subset=['RMSE', 'MAE'], color='#d4edda')
                    .highlight_max(subset=['R²'],          color='#d4edda')
                    .format({'RMSE': '{:.2f}', 'MAE': '{:.2f}', 'R²': '{:.4f}'}),
                use_container_width=True, hide_index=True
            )

    # Bảng tổng hợp
    st.markdown("---")
    st.subheader("Bảng tổng hợp")
    summary_rows = []
    for model_name, y_pred in predictions.items():
        for i, h in enumerate(HORIZONS):
            err  = Y_true[:, i] - y_pred[:, i]
            rmse = np.sqrt(np.mean(err**2))
            mae  = np.mean(np.abs(err))
            r2   = 1 - np.sum(err**2) / np.sum((Y_true[:, i] - np.mean(Y_true[:, i]))**2)
            summary_rows.append({'Model': model_name, 'Horizon': h,
                                  'RMSE': round(rmse, 2), 'MAE': round(mae, 2),
                                  'R²': round(r2, 4)})
    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

# ── Tab 2: Actual vs Predicted ────────────────
with tab2:
    st.subheader(f"Actual vs Predicted — {horizon_choice} ({n_display} giờ đầu tập Test)")
    fig = plot_actual_vs_pred(Y_true, predictions, horizon_idx, n_display)
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    **Cách đọc biểu đồ:**
    - Đường xám = giá trị thực tế
    - Đường màu = giá trị dự báo
    - Vùng tô nhạt = khoảng sai số
    - **Overfit**: train tốt nhưng đường dự báo lệch xa thực tế → R² thấp hơn nhiều so với Dongsi
    - **Underfit**: đường dự báo phẳng, không bắt được đỉnh/đáy của PM2.5
    """)

# ── Tab 3: Scatter ────────────────────────────
with tab3:
    st.subheader(f"Scatter Plot — {horizon_choice}")
    fig = plot_scatter(Y_true, predictions, horizon_idx)
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    **Cách đọc:**
    - Các chấm càng nằm sát đường đứt y=x → dự báo càng chính xác
    - Chấm phân tán ra xa → sai số lớn
    - Nếu chấm lệch hẳn về 1 phía → model có bias (luôn dự báo cao hoặc thấp hơn thực tế)
    """)