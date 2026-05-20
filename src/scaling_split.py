import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def scale_and_split(df):
    df = df.copy()

    #Tách X và Y
    target_cols = ['pm25_t1', 'pm25_t2', 'pm25_t3']
    drop_cols = ['datetime', 'PM2.5'] + target_cols
    X = df.drop(columns = drop_cols)
    Y = df[target_cols]

    #Chia train/test theo thơi fgian
    split_idx = int(len(df) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    Y_train = Y.iloc[:split_idx]
    Y_test = Y.iloc[split_idx:]

    #scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"train {X_train_scaled.shape}, test {X_test_scaled}")
    return X_train_scaled, X_test_scaled, Y_train, Y_test
