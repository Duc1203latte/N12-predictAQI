import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_cv_results(X_train_scaled, Y_train, tscv):
    """Chạy GridSearchCV và trả về object để lấy cv_results_"""
    param_grid = {'estimator__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    base_model = MultiOutputRegressor(Lasso(max_iter=100000, tol=1e-3), n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, Y_train)
    return grid_search
#ĐỌC DATA
if __name__ == "__main__":
    df = pd.read_csv("../data/processed/features_output1.csv", parse_dates=['datetime'])

    print("=" * 55)
    print("  DỮ LIỆU")
    print("=" * 55)
    print(f"  Shape: {df.shape[0]:,} dòng × {df.shape[1]} cột")

    #TÁCH X VÀ Y
    TARGET_COLS = ['pm25_t1', 'pm25_t2', 'pm25_t3']
    X = df.drop(columns=['datetime', 'PM2.5'] + TARGET_COLS)
    Y = df[TARGET_COLS]

    #CHRONOLOGICAL SPLIT 80/20
    split_idx = int(len(df) * 0.8)
    X_train   = X[:split_idx];  X_test  = X[split_idx:]
    Y_train   = Y[:split_idx];  Y_test  = Y[split_idx:]

    train_start = df['datetime'].iloc[0]
    train_end   = df['datetime'].iloc[split_idx - 1]
    test_start  = df['datetime'].iloc[split_idx]
    test_end    = df['datetime'].iloc[-1]

    print(f"\n  TRAIN : {X_train.shape[0]:,} dòng")
    print(f"    Từ  : {train_start.strftime('%Y-%m-%d')}")
    print(f"    Đến : {train_end.strftime('%Y-%m-%d')}")
    print(f"\n  TEST  : {X_test.shape[0]:,} dòng")
    print(f"    Từ  : {test_start.strftime('%Y-%m-%d')}")
    print(f"    Đến : {test_end.strftime('%Y-%m-%d')}")

    #MINMAXSCALER
    scaler         = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"\n  Scale xong — X_train_scaled: {X_train_scaled.shape}")

    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = get_cv_results(X_train_scaled, Y_train, tscv)

    best_model = grid_search.best_estimator_
    best_alpha = grid_search.best_params_['estimator__alpha']
    best_rmse  = abs(grid_search.best_score_)

    #KẾT QUẢ GRIDSEARCH
    print("\n" + "=" * 55)
    print("  KẾT QUẢ GRIDSEARCHCV")
    print("=" * 55)
    print(f"  Alpha tốt nhất : {best_alpha}")
    print(f"  RMSE tốt nhất  : {best_rmse:.4f} µg/m³ (TB 5-fold CV)")

    print(f"\n  {'Alpha':>10}  {'RMSE (CV)':>12}")
    print(f"  {'-'*10}  {'-'*12}")
    results = grid_search.cv_results_
    for alpha, score in zip(
        results['param_estimator__alpha'],
        results['mean_test_score']
    ):
        marker = '  <- best' if alpha == best_alpha else ''
        print(f"  {alpha:>10}  {abs(score):>10.4f}{marker}")

    #ĐÁNH GIÁ TRÊN TẬP TEST
    Y_pred = best_model.predict(X_test_scaled)

    print("\n" + "=" * 55)
    print("  ĐÁNH GIÁ TRÊN TẬP TEST")
    print("=" * 55)
    print(f"  {'Horizon':<10} {'RMSE':>10} {'MAE':>10} {'R2':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for i, horizon in enumerate(['t+1', 't+2', 't+3']):
        rmse = np.sqrt(mean_squared_error(Y_test.iloc[:, i], Y_pred[:, i]))
        mae  = mean_absolute_error(Y_test.iloc[:, i], Y_pred[:, i])
        r2   = r2_score(Y_test.iloc[:, i], Y_pred[:, i])
        print(f"  {horizon:<10} {rmse:>10.2f} {mae:>10.2f} {r2:>8.4f}")

    #THÔNG TIN FEATURES
    print("\n" + "=" * 55)
    print("  PHÂN TÍCH LASSO (model t+1)")
    print("=" * 55)
    coef       = best_model.estimators_[0].coef_
    n_zero     = np.sum(coef == 0)
    n_nonzero  = np.sum(coef != 0)
    print(f"  Tổng features  : {len(coef)}")
    print(f"  Giữ lại        : {n_nonzero}  (coef != 0)")
    print(f"  Bị loại        : {n_zero}  (coef = 0)")

    #LƯU MODEL
    os.makedirs('../models', exist_ok=True)
    joblib.dump(best_model, '../models/lasso_model.pkl')
    joblib.dump(scaler, '../models/lasso_scaler.pkl')

    print("\n" + "=" * 55)
    print("  ĐÃ LƯU")
    print("=" * 55)
    print(f"  models/lasso_model.pkl")
    print(f"  models/lasso_scaler.pkl")
    print("\n  HOÀN THANH!")
    print("=" * 55)