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
    page_title = "PM2.5 Forecast",
    page_icon  = "🌫️",
    layout     = "wide"
)

HORIZONS    = ['t+1', 't+2', 't+3']
TARGET_COLS = ['pm25_t1', 'pm25_t2', 'pm25_t3']
MODEL_COLORS = {'Lasso': '#2E75B6', 'Ridge': '#ED7D31', 'RF': '#70AD47'}

# AQI theo tiêu chuẩn Trung Quốc (µg/m³)
def aqi_level(pm25):
    if pm25 <= 35:   return "Tốt 🟢",        "#d4edda", "#155724"
    elif pm25 <= 75: return "Trung bình 🟡",  "#fff3cd", "#856404"
    elif pm25 <= 115:return "Kém 🟠",         "#ffe5d0", "#a04000"
    elif pm25 <= 150:return "Xấu 🔴",         "#f8d7da", "#721c24"
    else:            return "Nguy hại ⚫",    "#e2d5f8", "#3b0764"

# =============================================
# LOAD 3 MODEL
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
        'RF'   : (rf,    None),
    }

def calc_metrics(y_true, y_pred):
    rows = []
    for i, h in enumerate(HORIZONS):
        err  = y_true[:, i] - y_pred[:, i]
        rmse = np.sqrt(np.mean(err**2))
        mae  = np.mean(np.abs(err))
        ss_res = np.sum(err**2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i]))**2)
        r2 = 1 - ss_res / ss_tot
        rows.append({'Horizon': h, 'RMSE': round(rmse,2),
                     'MAE': round(mae,2), 'R²': round(r2,4)})
    return pd.DataFrame(rows)

def plot_actual_vs_pred(y_true, predictions, horizon_idx=0, n=300):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f'Actual vs Predicted — {HORIZONS[horizon_idx]}',
                 fontsize=13, fontweight='bold')
    for ax, (model_name, y_pred) in zip(axes, predictions.items()):
        color = MODEL_COLORS[model_name]
        ax.plot(range(n), y_true[:n, horizon_idx], color='#404040',
                linewidth=1.2, label='Actual', zorder=3)
        ax.plot(range(n), y_pred[:n, horizon_idx], color=color,
                linewidth=1.0, label=f'{model_name}', alpha=0.85)
        ax.fill_between(range(n), y_true[:n, horizon_idx],
                        y_pred[:n, horizon_idx], alpha=0.1, color=color)
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
st.title("🌫️ PM2.5 Forecast System")

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")
    uploaded_file = st.file_uploader(
        "Upload file CSV (Beijing Multisite)",
        type=["csv"],
        help="Ví dụ: PRSA_Data_Wanliu_20130301-20170228.csv"
    )
    st.markdown("---")
    st.markdown("**Model đã train:** Dongsi 2013–2017")
    st.markdown("**Dự báo:** PM2.5 tại t+1, t+2, t+3")

if uploaded_file is None:
    st.info("👈 Upload file CSV ở thanh bên trái để bắt đầu.")
    st.markdown("""
    **Các trạm có thể dùng:**
    `Aotizhongxin` · `Changping` · `Dingling` · `Dongsi` · `Guanyuan`
    `Gucheng` · `Huairou` · `Nongzhanguan` · `Shunyi` · `Tiantan`
    `Wanliu` · `Wanshouxigong`
    """)
    st.stop()

# ── Preprocessing ────────────────────────────
with st.spinner("Đang xử lý dữ liệu..."):
    try:
        tmp_path = f"data/raw/{uploaded_file.name}"
        os.makedirs("data/raw", exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        df_clean = clean_data(tmp_path)
        df_feat  = build_features(df_clean)
        station_name = uploaded_file.name.replace("PRSA_Data_","").split("_")[0]
        st.success(f"✅ Xử lý xong — Trạm: **{station_name}** | "
                   f"{df_feat.shape[0]:,} dòng | {df_feat.shape[1]} cột")
    except Exception as e:
        st.error(f"❌ Lỗi preprocessing: {e}")
        st.stop()

X       = df_feat.drop(columns=['datetime', 'PM2.5'] + TARGET_COLS)
Y       = df_feat[TARGET_COLS]
models  = load_models()

# ── 3 TABS ───────────────────────────────────
tab_forecast, tab_eval, tab_chart = st.tabs([
    "🔮 Dự báo", "📊 Đánh giá model", "📈 Biểu đồ"
])

# ══════════════════════════════════════════════
# TAB 1 — DỰ BÁO
# ══════════════════════════════════════════════
with tab_forecast:
    st.subheader(f"Dự báo PM2.5 — Trạm {station_name}")
    st.markdown("Chọn thời điểm bất kỳ trong dữ liệu, hệ thống sẽ dự báo PM2.5 cho 3 giờ tiếp theo.")

    # Chọn thời điểm
    dt_list     = df_feat['datetime'].dt.to_pydatetime().tolist()
    chosen_dt   = st.select_slider(
        "Chọn thời điểm dự báo",
        options=dt_list,
        value=dt_list[int(len(dt_list)*0.8)],   # mặc định đầu tập test
        format_func=lambda x: x.strftime("%Y-%m-%d %H:00")
    )

    # Lấy row tương ứng
    row_idx = df_feat[df_feat['datetime'] == pd.Timestamp(chosen_dt)].index
    if len(row_idx) == 0:
        st.warning("Không tìm thấy dữ liệu tại thời điểm này.")
        st.stop()

    row_idx = row_idx[0]
    X_row   = X.loc[[row_idx]]                # features tại thời điểm đó
    Y_row   = Y.loc[row_idx]                  # nhãn thực tế (nếu có)
    pm25_now= df_feat.loc[row_idx, 'PM2.5']   # PM2.5 hiện tại

    st.markdown(f"**Thời điểm:** {chosen_dt.strftime('%Y-%m-%d %H:00')} &nbsp;|&nbsp; "
                f"**PM2.5 hiện tại:** {pm25_now:.1f} µg/m³")
    st.markdown("---")

    # Dự báo với cả 3 model
    st.markdown("### Kết quả dự báo")
    cols = st.columns(3)

    for col, horizon, h_label in zip(cols, TARGET_COLS, HORIZONS):
        with col:
            st.markdown(f"**{h_label}**")
            model_cols = st.columns(3)
            for mc, (model_name, (model, scaler)) in zip(model_cols, models.items()):
                X_scaled = scaler.transform(X_row) if scaler else X_row
                pred     = model.predict(X_scaled)[0]
                pred_val = pred[TARGET_COLS.index(horizon)]
                label, bg, fg = aqi_level(pred_val)
                mc.markdown(
                    f"<div style='background:{bg};color:{fg};padding:10px;"
                    f"border-radius:8px;text-align:center'>"
                    f"<b>{model_name}</b><br>"
                    f"<span style='font-size:22px;font-weight:bold'>{pred_val:.1f}</span>"
                    f" µg/m³<br><small>{label}</small></div>",
                    unsafe_allow_html=True
                )

    # So sánh với thực tế
    st.markdown("---")
    st.markdown("### So sánh với thực tế")
    actual_cols = st.columns(3)
    for col, horizon, h_label in zip(actual_cols, TARGET_COLS, HORIZONS):
        actual_val = Y_row[horizon]
        label, bg, fg = aqi_level(actual_val)
        col.markdown(
            f"<div style='background:{bg};color:{fg};padding:10px;"
            f"border-radius:8px;text-align:center'>"
            f"<b>Thực tế {h_label}</b><br>"
            f"<span style='font-size:22px;font-weight:bold'>{actual_val:.1f}</span>"
            f" µg/m³<br><small>{label}</small></div>",
            unsafe_allow_html=True
        )

    # Bảng AQI tham khảo
    with st.expander("📖 Bảng mức độ ô nhiễm PM2.5 (tiêu chuẩn Trung Quốc)"):
        st.markdown("""
        | Mức | PM2.5 (µg/m³) | Ý nghĩa |
        |---|---|---|
        | 🟢 Tốt | 0 – 35 | Không khí trong lành |
        | 🟡 Trung bình | 35 – 75 | Nhóm nhạy cảm nên hạn chế ra ngoài |
        | 🟠 Kém | 75 – 115 | Hạn chế hoạt động ngoài trời |
        | 🔴 Xấu | 115 – 150 | Tránh ra ngoài |
        | ⚫ Nguy hại | > 150 | Ở trong nhà, đóng cửa sổ |
        """)

# ══════════════════════════════════════════════
# TAB 2 — ĐÁNH GIÁ MODEL
# ══════════════════════════════════════════════
with tab_eval:
    split_ratio = st.slider("Tỉ lệ Train/Test", 0.6, 0.9, 0.8, 0.05)
    split_idx   = int(len(df_feat) * split_ratio)
    X_test      = X.iloc[split_idx:]
    Y_test      = Y.iloc[split_idx:]
    Y_true      = Y_test.values

    predictions = {}
    for model_name, (model, scaler) in models.items():
        X_scaled = scaler.transform(X_test) if scaler else X_test
        predictions[model_name] = model.predict(X_scaled)

    st.subheader(f"So sánh 3 model — Trạm {station_name}")
    cols = st.columns(3)
    for col, (model_name, y_pred) in zip(cols, predictions.items()):
        df_m  = calc_metrics(Y_true, y_pred)
        color = MODEL_COLORS[model_name]
        with col:
            st.markdown(f"<h3 style='color:{color}'>{model_name}</h3>",
                        unsafe_allow_html=True)
            st.dataframe(
                df_m.style
                    .highlight_min(subset=['RMSE','MAE'], color='#d4edda')
                    .highlight_max(subset=['R²'],         color='#d4edda')
                    .format({'RMSE':'{:.2f}','MAE':'{:.2f}','R²':'{:.4f}'}),
                use_container_width=True, hide_index=True
            )

    st.markdown("---")
    st.subheader("Bảng tổng hợp")
    rows = []
    for model_name, y_pred in predictions.items():
        for i, h in enumerate(HORIZONS):
            err  = Y_true[:,i] - y_pred[:,i]
            rmse = np.sqrt(np.mean(err**2))
            mae  = np.mean(np.abs(err))
            r2   = 1 - np.sum(err**2)/np.sum((Y_true[:,i]-np.mean(Y_true[:,i]))**2)
            rows.append({'Model':model_name,'Horizon':h,
                         'RMSE':round(rmse,2),'MAE':round(mae,2),'R²':round(r2,4)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 3 — BIỂU ĐỒ
# ══════════════════════════════════════════════
with tab_chart:
    col1, col2 = st.columns(2)
    with col1:
        horizon_choice = st.selectbox("Horizon", HORIZONS, index=0)
    with col2:
        n_display = st.slider("Số giờ hiển thị", 100, 500, 300, 50)

    horizon_idx = HORIZONS.index(horizon_choice)

    # Dùng lại predictions từ tab 2 (split mặc định 80/20)
    split_idx   = int(len(df_feat) * 0.8)
    X_test      = X.iloc[split_idx:]
    Y_test      = Y.iloc[split_idx:]
    Y_true      = Y_test.values
    predictions = {}
    for model_name, (model, scaler) in models.items():
        X_scaled = scaler.transform(X_test) if scaler else X_test
        predictions[model_name] = model.predict(X_scaled)

    st.markdown("#### Actual vs Predicted")
    fig = plot_actual_vs_pred(Y_true, predictions, horizon_idx, n_display)
    st.pyplot(fig); plt.close()

    st.markdown("#### Scatter Plot")
    fig = plot_scatter(Y_true, predictions, horizon_idx)
    st.pyplot(fig); plt.close()