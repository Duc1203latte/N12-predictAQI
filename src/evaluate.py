import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import os

# =============================================
# CẤU HÌNH
# =============================================
plt.rcParams['font.family']     = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

COLORS = {
    'lasso': '#2E75B6',
    'ridge': '#ED7D31',
    'rf'   : '#70AD47',
}
HORIZONS = ['t+1', 't+2', 't+3']
os.makedirs('outputs/charts', exist_ok=True)

# =============================================
# 1. NẠP DỮ LIỆU VÀ DỰ BÁO
# =============================================
print("Đang nạp dữ liệu...")
df = pd.read_csv("data/processed/features_output1.csv", parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

TARGET_COLS = ['pm25_t1', 'pm25_t2', 'pm25_t3']
X = df.drop(columns=['datetime', 'PM2.5'] + TARGET_COLS)
Y = df[TARGET_COLS]

split_idx  = int(len(df) * 0.8)
X_test     = X.iloc[split_idx:]
Y_test     = Y.iloc[split_idx:]
dt_test    = df['datetime'].iloc[split_idx:].reset_index(drop=True)

# Nạp 3 model và dự báo
print("Đang nạp model và dự báo...")

lasso_model  = joblib.load('models/lasso_model.pkl')
lasso_scaler = joblib.load('models/lasso_scaler.pkl')
X_test_lasso = lasso_scaler.transform(X_test)
Y_pred_lasso = lasso_model.predict(X_test_lasso)

ridge_model  = joblib.load('models/ridge_model.pkl')
ridge_scaler = joblib.load('models/ridge_scaler.pkl')
X_test_ridge = ridge_scaler.transform(X_test)
Y_pred_ridge = ridge_model.predict(X_test_ridge)

rf_model     = joblib.load('models/random_forest_model.pkl')
Y_pred_rf    = rf_model.predict(X_test)   # RF không cần scale

Y_true = Y_test.values

# =============================================
# TÍNH METRICS
# =============================================
def calc_metrics(y_true, y_pred):
    # Trả về dict {horizon: {rmse, mae, r2}}
    results = {}
    for i, h in enumerate(HORIZONS):
        err  = y_true[:, i] - y_pred[:, i]
        rmse = np.sqrt(np.mean(err**2))
        mae  = np.mean(np.abs(err))
        ss_res = np.sum(err**2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i]))**2)
        r2 = 1 - ss_res / ss_tot
        results[h] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    return results

metrics = {
    'Lasso': calc_metrics(Y_true, Y_pred_lasso),
    'Ridge': calc_metrics(Y_true, Y_pred_ridge),
    'RF'   : calc_metrics(Y_true, Y_pred_rf),
}

# =============================================
# BIỂU ĐỒ 1 — SO SÁNH RMSE & MAE & R²
# =============================================
print("Đang vẽ Biểu đồ 1: So sánh metrics...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('So sánh 3 Model — RMSE, MAE, R²', fontsize=14, fontweight='bold', y=1.02)

metric_names = ['RMSE', 'MAE', 'R2']
y_labels     = ['RMSE (µg/m³)', 'MAE (µg/m³)', 'R²']
model_names  = ['Lasso', 'Ridge', 'RF']
bar_colors   = [COLORS['lasso'], COLORS['ridge'], COLORS['rf']]

x     = np.arange(len(HORIZONS))  # vị trí horizon trên trục x
width = 0.25                       # độ rộng mỗi cột

for ax_idx, (metric, ylabel) in enumerate(zip(metric_names, y_labels)):
    ax = axes[ax_idx]
    for m_idx, (model, color) in enumerate(zip(model_names, bar_colors)):
        # Lấy giá trị metric của model này qua 3 horizon
        vals = [metrics[model][h][metric] for h in HORIZONS]
        bars = ax.bar(x + m_idx * width, vals, width,
                      label=model, color=color, alpha=0.85, edgecolor='white')
        # Ghi số lên đầu cột
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.3 if metric != 'R2' else 0.003),
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Horizon', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(HORIZONS)
    ax.legend(fontsize=9)
    if metric == 'R2':
        ax.set_ylim(0.75, 1.01)   # zoom vào vùng khác biệt
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/charts/01_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> Đã lưu: outputs/charts/01_metrics_comparison.png")

# =============================================
# BIỂU ĐỒ 2 — ACTUAL vs PREDICTED (t+1, 200 giờ đầu test)
# =============================================
print("Đang vẽ Biểu đồ 2: Actual vs Predicted...")

N = 200   # số giờ hiển thị (200 giờ đầu của test)
fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
fig.suptitle('Actual vs Predicted — PM2.5 t+1 (200 giờ đầu tập Test)', fontsize=13, fontweight='bold')

for ax_idx, (model_name, y_pred, color) in enumerate([
    ('Lasso', Y_pred_lasso, COLORS['lasso']),
    ('Ridge', Y_pred_ridge, COLORS['ridge']),
    ('RF',    Y_pred_rf,    COLORS['rf']),
]):
    ax = axes[ax_idx]
    ax.plot(range(N), Y_true[:N, 0],   color='#404040', linewidth=1.2, label='Actual', zorder=3)
    ax.plot(range(N), y_pred[:N, 0],   color=color, linewidth=1.0, label=f'{model_name} Predicted', alpha=0.85, zorder=2)
    ax.fill_between(range(N), Y_true[:N, 0], y_pred[:N, 0], alpha=0.12, color=color)  # tô vùng sai số
    ax.set_ylabel('PM2.5 (µg/m³)', fontsize=10)
    ax.set_title(f'{model_name}  —  MAE={metrics[model_name]["t+1"]["MAE"]:.2f}, R²={metrics[model_name]["t+1"]["R2"]:.4f}', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)

axes[-1].set_xlabel('Giờ (tập Test)', fontsize=10)
plt.tight_layout()
plt.savefig('outputs/charts/02_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> Đã lưu: outputs/charts/02_actual_vs_predicted.png")

# =============================================
# BIỂU ĐỒ 3 — SCATTER: Actual vs Predicted (t+1)
# =============================================
print("Đang vẽ Biểu đồ 3: Scatter plot...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Scatter: Actual vs Predicted — t+1', fontsize=13, fontweight='bold')

for ax, (model_name, y_pred, color) in zip(axes, [
    ('Lasso', Y_pred_lasso, COLORS['lasso']),
    ('Ridge', Y_pred_ridge, COLORS['ridge']),
    ('RF',    Y_pred_rf,    COLORS['rf']),
]):
    ax.scatter(Y_true[:, 0], y_pred[:, 0],
               color=color, alpha=0.25, s=6, edgecolors='none')
    # Đường lý tưởng y=x
    lim = max(Y_true[:, 0].max(), y_pred[:, 0].max())
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1, label='Lý tưởng (y=x)')
    ax.set_xlabel('Actual PM2.5 (µg/m³)', fontsize=10)
    ax.set_ylabel('Predicted PM2.5 (µg/m³)', fontsize=10)
    ax.set_title(f'{model_name}  R²={metrics[model_name]["t+1"]["R2"]:.4f}', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('outputs/charts/03_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> Đã lưu: outputs/charts/03_scatter.png")

# =============================================
# BIỂU ĐỒ 4 — FEATURE IMPORTANCE (RF, top 15)
# =============================================
print("Đang vẽ Biểu đồ 4: Feature Importance...")

importances   = rf_model.feature_importances_   # mảng độ quan trọng của từng feature
feature_names = X.columns.tolist()               # tên các feature

top_n    = 15
top_idx  = np.argsort(importances)[::-1][:top_n]    # sắp xếp giảm dần, lấy top 15
top_vals = importances[top_idx]
top_names= [feature_names[i] for i in top_idx]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(range(top_n), top_vals[::-1],   # đảo ngược để feature quan trọng nhất ở trên cùng
               color=COLORS['rf'], alpha=0.85, edgecolor='white')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_names[::-1], fontsize=10)
ax.set_xlabel('Importance Score', fontsize=11)
ax.set_title('Top 15 Feature Importance — Random Forest', fontsize=13, fontweight='bold')

# Ghi số cuối mỗi thanh
for bar, v in zip(bars, top_vals[::-1]):
    ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{v:.4f}', va='center', fontsize=8)

ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/charts/04_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> Đã lưu: outputs/charts/04_feature_importance.png")

print("\n✅ Xong! Tất cả biểu đồ đã lưu vào thư mục outputs/charts/")