# train_lasso.py — thay toàn bộ phần đầu thành:

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler          # ← dùng MinMax thay Standard
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv("data/processed/features_output1.csv", parse_dates=['datetime'])

TARGET_COLS = ['pm25_t1', 'pm25_t2', 'pm25_t3']
X = df.drop(columns=['datetime'] + TARGET_COLS)
Y = df[TARGET_COLS]

# ── Chronological split ──────────────────────────────────────
split_idx      = int(len(df) * 0.8)
X_train        = X[:split_idx];  X_test  = X[split_idx:]
Y_train        = Y[:split_idx];  Y_test  = Y[split_idx:]

# ── MinMaxScaler — Lasso hội tụ tốt hơn với [0,1] ───────────
scaler         = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train — alpha=0.01 đã tìm được từ GridSearch trước ──────
best_model = MultiOutputRegressor(
    Lasso(alpha=0.01, max_iter=100000, tol=1e-3),
    n_jobs=-1
)
best_model.fit(X_train_scaled, Y_train)

# ── Đánh giá ─────────────────────────────────────────────────
Y_pred = best_model.predict(X_test_scaled)

print("\n--- Lasso Model Evaluation ---")
for i, horizon in enumerate(['t+1', 't+2', 't+3']):
    rmse = np.sqrt(mean_squared_error(Y_test.iloc[:, i], Y_pred[:, i]))
    mae  = mean_absolute_error(Y_test.iloc[:, i], Y_pred[:, i])
    r2   = r2_score(Y_test.iloc[:, i], Y_pred[:, i])
    print(f"{horizon}  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")

# ── Lưu model ────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/lasso_model.pkl')
joblib.dump(scaler,     'models/lasso_scaler.pkl')   # lưu riêng để phân biệt
print("\nĐã lưu: models/lasso_model.pkl + lasso_scaler.pkl")