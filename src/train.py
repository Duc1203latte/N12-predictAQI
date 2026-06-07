import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import os

# Import hàm từ 3 file train
from train_lasso import get_cv_results as lasso_cv_results
from train_ridge import get_cv_results as ridge_cv_results
from train_rf    import get_learning_curve

# =============================================
# CẤU HÌNH
# =============================================
plt.rcParams['font.family']       = 'DejaVu Sans'
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

COLORS     = {'lasso': '#2E75B6', 'ridge': '#ED7D31', 'rf': '#70AD47'}
ALPHAS     = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
N_TREES    = [5, 10, 20, 30, 50, 75, 100, 150, 200]
os.makedirs('outputs/charts', exist_ok=True)

# =============================================
# CHUẨN BỊ DỮ LIỆU
# =============================================
print("Đang nạp dữ liệu...")
df = pd.read_csv("data/processed/features_output1.csv", parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

TARGET_COLS = ['pm25_t1', 'pm25_t2', 'pm25_t3']
X = df.drop(columns=['datetime', 'PM2.5'] + TARGET_COLS)
Y = df[TARGET_COLS]

split_idx  = int(len(df) * 0.8)
X_train    = X.iloc[:split_idx];  X_test  = X.iloc[split_idx:]
Y_train    = Y.iloc[:split_idx];  Y_test  = Y.iloc[split_idx:]

# Scale cho Lasso và Ridge (RF không cần)
scaler         = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

tscv = TimeSeriesSplit(n_splits=5)

# =============================================
# GỌI CÁC HÀM TỪ FILE TRAIN
# =============================================
print("\nĐang chạy Lasso GridSearchCV...")
lasso_gs = lasso_cv_results(X_train_scaled, Y_train, tscv)
lasso_cv_mean = [abs(s) for s in lasso_gs.cv_results_['mean_test_score']]
lasso_cv_std  = list(lasso_gs.cv_results_['std_test_score'])
best_lasso_alpha = lasso_gs.best_params_['estimator__alpha']

print("\nĐang chạy Ridge GridSearchCV...")
ridge_gs = ridge_cv_results(X_train_scaled, Y_train, tscv)
ridge_cv_mean = [abs(s) for s in ridge_gs.cv_results_['mean_test_score']]
ridge_cv_std  = list(ridge_gs.cv_results_['std_test_score'])
best_ridge_alpha = ridge_gs.best_params_['estimator__alpha']

print("\nĐang chạy RF learning curve...")
rf_train_rmse, rf_test_rmse = get_learning_curve(X_train, Y_train, X_test, Y_test, N_TREES)
best_n = N_TREES[np.argmin(rf_test_rmse)]

# =============================================
# VẼ BIỂU ĐỒ
# =============================================
print("\nĐang vẽ biểu đồ...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Learning Curve — Hiệu năng theo Tham số', fontsize=14, fontweight='bold', y=1.02)

# ── Subplot 1: Lasso ──
ax = axes[0]
ax.semilogx(ALPHAS, lasso_cv_mean,
            color=COLORS['lasso'], marker='o', linewidth=2, markersize=7, label='CV RMSE (mean)')
ax.fill_between(ALPHAS,
                np.array(lasso_cv_mean) - np.array(lasso_cv_std),
                np.array(lasso_cv_mean) + np.array(lasso_cv_std),
                alpha=0.15, color=COLORS['lasso'], label='±1 std')
best_idx = np.argmin(lasso_cv_mean)
ax.axvline(x=best_lasso_alpha, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
ax.scatter([best_lasso_alpha], [lasso_cv_mean[best_idx]], color='red', zorder=5, s=80)
ax.text(best_lasso_alpha * 1.3, lasso_cv_mean[best_idx] + 0.5,
        f'Best α={best_lasso_alpha}', color='red', fontsize=9)
ax.set_xscale('log')
ax.set_xlabel('Alpha (log scale)', fontsize=11)
ax.set_ylabel('RMSE (µg/m³)', fontsize=11)
ax.set_title('Lasso — RMSE theo Alpha', fontsize=12, fontweight='bold', color=COLORS['lasso'])
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── Subplot 2: Ridge ──
ax = axes[1]
ax.semilogx(ALPHAS, ridge_cv_mean,
            color=COLORS['ridge'], marker='s', linewidth=2, markersize=7, label='CV RMSE (mean)')
ax.fill_between(ALPHAS,
                np.array(ridge_cv_mean) - np.array(ridge_cv_std),
                np.array(ridge_cv_mean) + np.array(ridge_cv_std),
                alpha=0.15, color=COLORS['ridge'], label='±1 std')
best_idx = np.argmin(ridge_cv_mean)
ax.axvline(x=best_ridge_alpha, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
ax.scatter([best_ridge_alpha], [ridge_cv_mean[best_idx]], color='red', zorder=5, s=80)
ax.text(best_ridge_alpha * 1.3, ridge_cv_mean[best_idx] + 0.5,
        f'Best α={best_ridge_alpha}', color='red', fontsize=9)
ax.set_xscale('log')
ax.set_xlabel('Alpha (log scale)', fontsize=11)
ax.set_ylabel('RMSE (µg/m³)', fontsize=11)
ax.set_title('Ridge — RMSE theo Alpha', fontsize=12, fontweight='bold', color=COLORS['ridge'])
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── Subplot 3: Random Forest ──
ax = axes[2]
ax.plot(N_TREES, rf_train_rmse,
        color='gray', marker='^', linewidth=2, markersize=7,
        linestyle='--', label='Train RMSE', alpha=0.8)
ax.plot(N_TREES, rf_test_rmse,
        color=COLORS['rf'], marker='o', linewidth=2, markersize=7, label='Test RMSE')
ax.axvline(x=best_n, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
ax.scatter([best_n], [min(rf_test_rmse)], color='red', zorder=5, s=80)
ax.text(best_n + 3, min(rf_test_rmse) + 0.3, f'Best n={best_n}', color='red', fontsize=9)
ax.set_xlabel('Số cây (n_estimators)', fontsize=11)
ax.set_ylabel('RMSE (µg/m³)', fontsize=11)
ax.set_title('Random Forest — RMSE theo số cây', fontsize=12, fontweight='bold', color=COLORS['rf'])
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/charts/05_train_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ Đã lưu: outputs/charts/05_train_curve.png")