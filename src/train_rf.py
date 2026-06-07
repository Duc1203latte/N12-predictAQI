import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def get_learning_curve(X_train, Y_train, X_test, Y_test,
                       n_trees_list=[5, 10, 20, 30, 50, 75, 100, 150, 200]):
    """Train RF với nhiều n_estimators, trả về train/test RMSE để vẽ đồ thị"""
    train_rmse_list = []
    test_rmse_list = []

    for n in n_trees_list:
        model = RandomForestRegressor(n_estimators=n, n_jobs=-1, random_state=42)
        model.fit(X_train, Y_train)

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        train_rmse = np.sqrt(np.mean((Y_train.values - pred_train) ** 2))
        test_rmse = np.sqrt(np.mean((Y_test.values - pred_test) ** 2))

        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
        print(f"  n_estimators={n:<5} Train RMSE={train_rmse:.4f}  Test RMSE={test_rmse:.4f}")

    return train_rmse_list, test_rmse_list

if __name__ == "__main__":
    print("=== GIAI ĐOẠN 1: NẠP DỮ LIỆU ĐÃ XỬ LÝ ===")
    data_path = "../data/processed/features_output1.csv"
    df = pd.read_csv(data_path)

    # Xác định danh sách nhãn mục tiêu (Target Y)
    target_cols = ['pm25_t1', 'pm25_t2', 'pm25_t3']

    # Tách ma trận đặc trưng X và nhãn Y
    X = df.drop(columns=target_cols + ['datetime', 'PM2.5'])  # thêm 'PM2.5' vào drop
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

    print("\n=== GIAI ĐOẠN 4: ĐÁNH GIÁ TRÊN TẬP TEST ===")  # thêm phần đánh giá
    Y_pred = rf_model.predict(X_test)

    print(f"\n  {'Horizon':<10} {'RMSE':>10} {'MAE':>10} {'R2':>8}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}")

    for i, horizon in enumerate(['t+1', 't+2', 't+3']):
        rmse = np.sqrt(mean_squared_error(Y_test.iloc[:, i], Y_pred[:, i]))
        mae = mean_absolute_error(Y_test.iloc[:, i], Y_pred[:, i])
        r2 = r2_score(Y_test.iloc[:, i], Y_pred[:, i])
        print(f"  {horizon:<10} {rmse:>10.2f} {mae:>10.2f} {r2:>8.4f}")

    print("\n=== GIAI ĐOẠN 5: PHÂN TÍCH FEATURE IMPORTANCE ===")  # bonus RF
    importances = rf_model.feature_importances_
    feature_names = X.columns.tolist()

    top10_idx = np.argsort(importances)[::-1][:10]
    print(f"\n  Top 10 features quan trọng nhất:")
    print(f"  {'Feature':<25} {'Importance':>10}")
    print(f"  {'-' * 25} {'-' * 10}")
    for idx in top10_idx:
        print(f"  {feature_names[idx]:<25} {importances[idx]:>10.4f}")

    print("\n=== GIAI ĐOẠN 6: XUẤT MÔ HÌNH ĐẦU RA ===")
    os.makedirs('../models', exist_ok=True)  # tạo thư mục nếu chưa có
    joblib.dump(rf_model, '../models/random_forest_model.pkl')  # sửa đường dẫn

    print("Lưu thành công tại 'models/random_forest_model.pkl'")
    print("\nHOÀN THÀNH!")

