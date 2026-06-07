import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_cv_results(X_train_scaled, Y_train, tscv):
    """Chạy GridSearchCV và trả về object để lấy cv_results_"""
    param_grid = {'estimator__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    base_model = MultiOutputRegressor(Ridge(), n_jobs=-1)
    grid_search = GridSearchCV(
        estimator  = base_model,
        param_grid = param_grid,
        cv         = tscv,
        scoring    = 'neg_root_mean_squared_error',
        n_jobs     = -1
    )
    grid_search.fit(X_train_scaled, Y_train)
    return grid_search

if __name__ == "__main__":
    # 1. Nạp dữ liệu đã xử lý
    df = pd.read_csv("../data/processed/features_output1.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print("=" * 55)
    print("  DỮ LIỆU")
    print("=" * 55)
    print(f"  Shape: {df.shape[0]:,} dòng × {df.shape[1]} cột")

    # 2. Tách đặc trưng (X) và nhãn mục tiêu (y)
    TARGET_COLS = ['pm25_t1', 'pm25_t2', 'pm25_t3']
    X = df.drop(columns=['datetime', 'PM2.5'] + TARGET_COLS)
    Y = df[TARGET_COLS]


    # 3. Chia tập Train/Test theo chuỗi thời gian
    train_size = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    Y_train, Y_test = Y.iloc[:train_size], Y.iloc[train_size:]

    print(f"\n  TRAIN : {len(X_train):,} dòng")
    print(f"    Từ  : {df['datetime'].iloc[0].strftime('%Y-%m-%d')}")
    print(f"    Đến : {df['datetime'].iloc[train_size - 1].strftime('%Y-%m-%d')}")
    print(f"\n  TEST  : {len(X_test):,} dòng")
    print(f"    Từ  : {df['datetime'].iloc[train_size].strftime('%Y-%m-%d')}")
    print(f"    Đến : {df['datetime'].iloc[-1].strftime('%Y-%m-%d')}")

    # 4. Chuẩn hóa dữ liệu (Standardization)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n Scale xong - X_train_scaled: {X_train_scaled.shape}")

    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = get_cv_results(X_train_scaled, Y_train, tscv)

    best_model = grid_search.best_estimator_
    best_alpha = grid_search.best_params_['estimator__alpha']
    best_rmse  = abs(grid_search.best_score_)

    # 6. Kết quả GridSearch
    print("\n" + "=" * 55)
    print("  KẾT QUẢ GRIDSEARCHCV")
    print("=" * 55)
    print(f"  Alpha tốt nhất : {best_alpha}")
    print(f"  RMSE tốt nhất  : {best_rmse:.4f} µg/m³ (TB 5-fold CV)")

    print(f"\n  {'Alpha':>10}  {'RMSE (CV)':>12}")
    print(f"  {'-' * 10}  {'-' * 12}")
    results = grid_search.cv_results_
    for alpha, score in zip(
            results['param_estimator__alpha'],
            results['mean_test_score']
    ):
        marker = '  <- best' if alpha == best_alpha else ''
        print(f"  {alpha:>10}  {abs(score):>10.4f}{marker}")

    # 7. Đánh giá trên tập Test
    Y_pred = best_model.predict(X_test_scaled)

    print("\n" + "=" * 55)
    print("  ĐÁNH GIÁ TRÊN TẬP TEST")
    print("=" * 55)
    print(f"  {'Horizon':<10} {'RMSE':>10} {'MAE':>10} {'R2':>8}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}")

    for i, horizon in enumerate(['t+1', 't+2', 't+3']):
        rmse = np.sqrt(mean_squared_error(Y_test.iloc[:, i], Y_pred[:, i]))
        mae = mean_absolute_error(Y_test.iloc[:, i], Y_pred[:, i])
        r2 = r2_score(Y_test.iloc[:, i], Y_pred[:, i])
        print(f"  {horizon:<10} {rmse:>10.2f} {mae:>10.2f} {r2:>8.4f}")

    # 8. Lưu model
    os.makedirs('../models', exist_ok=True)
    joblib.dump(best_model, '../models/ridge_model.pkl')
    joblib.dump(scaler, '../models/ridge_scaler.pkl')

    print("\n" + "=" * 55)
    print("  ĐÃ LƯU")
    print("=" * 55)
    print(f"  models/ridge_model.pkl")
    print(f"  models/ridge_scaler.pkl")
    print("\n  HOÀN THÀNH!")
