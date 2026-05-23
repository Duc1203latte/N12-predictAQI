import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib

print("=== GIAI ĐOẠN 1: NẠP DỮ LIỆU ĐÃ XỬ LÝ ===")
data_path = "data/processed/features_output1.csv"
df = pd.read_csv(data_path)

# Xác định danh sách nhãn mục tiêu (Target Y)
target_cols = ['pm25_t1', 'pm25_t2', 'pm25_t3']

# Không dùng làm đặc trưng huấn luyện
ignored_cols = ['datetime']

# Tách ma trận đặc trưng X và nhãn Y
X = df.drop(columns=target_cols + ignored_cols)
Y = df[target_cols]

print(f"Tổng số lượng đặc trưng đầu vào (X): {X.shape[1]} cột")
print(f"Số lượng nhãn dự báo tương lai (Y): {Y.shape[1]} cột")

print("\n=== GIAI ĐOẠN 2: CHIA TẬP TRAIN/TEST ===")
# Chia tập chuỗi thời gian tuần tự
# 80% dữ liệu quá khứ để train
train_size = int(len(df) * 0.8)

X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
Y_train, Y_test = Y.iloc[:train_size], Y.iloc[train_size:]

print(f"Kích thước tập Train: {X_train.shape}")
print(f"Kích thước tập Test: {X_test.shape}")

print("\n=== GIAI ĐOẠN 3: HUẤN LUYỆN MÔ HÌNH RANDOM FOREST ===")
print("Running...")

# Khởi tạo mô hình Random Forest Regressor
# n_estimators=100: Xây dựng 100 cây quyết định
# n_jobs=-1: Kích hoạt tất cả các nhân/luồng
rf_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

# Huấn luyện mô hình
rf_model.fit(X_train, Y_train)
print("Huấn luyện thành công!!")

print("\n=== GIAI ĐOẠN 4: XUẤT MÔ HÌNH ĐẦU RA ===")
model_filename = "random_forest_model.pkl"

# Lưu mô hình
joblib.dump(rf_model, model_filename)

print(f"Lưu thành công tại file '{model_filename}'")
