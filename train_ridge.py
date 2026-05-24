import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Nạp dữ liệu đã xử lý
df = pd.read_csv("data/processed/features_output1.csv")

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# 2. Tách đặc trưng (X) và nhãn mục tiêu (y)
target_col = 'PM2.5'
drop_cols = ['datetime', target_col]

X = df.drop(columns=drop_cols)
y = df[target_col]

# 3. Chia tập Train/Test theo chuỗi thời gian
train_size = int(len(df) * 0.8)

X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# 4. Chuẩn hóa dữ liệu (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Huấn luyện mô hình Ridge Regression
ridge_model = Ridge(alpha=1.0) 
ridge_model.fit(X_train_scaled, y_train)

# 6. Dự đoán 
y_pred = ridge_model.predict(X_test_scaled)

# 7. Đánh giá mô hình
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE):     {mae:.2f}")
print(f"R-squared (R2) Score:         {r2:.4f}")
