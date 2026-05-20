import pandas as pd
import numpy as np

def build_features(df):
    df = df.copy()
    #nhóm 1: lag PM2.5
    for lag in [1,2,3,6,12,24]:
        df[f'pm25_lag_{lag}'] = df['PM2.5'].shift(lag) #lấy giá trị PM2.5 của lag giờ trước

    #Nhóm 2: lag khí tượng
    meteo_cols = ['TEMP', 'DEWP', 'PRES', 'WSPM', 'RAIN']
    for col in meteo_cols:
        for lag in [1,2,3]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    #Nhóm 3: lag chất ô nhiễm
    pollutant_cols = ['SO2', 'NO2', 'CO', 'O3', 'PM10']
    for col in pollutant_cols:
        for lag in [1,2,3]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    #Nhóm 4: rolling statistics
    df['pm25_rolling_mean_3h'] = df['PM2.5'].rolling(window=3).mean() #TB 3 giờ gần nhất
    df['pm25_rolling_mean_6h'] = df['PM2.5'].rolling(window=6).mean() #TB 6 giờ gân fnhaats
    df['pm25_rolling_std_6h'] = df['PM2.5'].rolling(window=6).std() #ộ lệch chuẩn trong 6 giờ gân fnhaats
    df['pm25_diff_1'] = df['PM2.5'].diff(1) #tốc độ thay đổi của t và t-1

    #Nhóm 5: biến thời gian hoàn toàn
    #trích xuất time từ cột datetime
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dow'] = df['datetime'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df = df.drop(columns=['hour', 'month', 'dow'])

    #Tạo nhãn Y của 3 giờ sau
    df['pm25_t1'] = df['PM2.5'].shift(-1)
    df['pm25_t2'] = df['PM2.5'].shift(-2)
    df['pm25_t3'] = df['PM2.5'].shift(-3)

    #Xóa dòng NAN
    df = df.dropna()
    df = df.drop(columns=['station'])
    print(f"Feature xong: {df.shape[0]} dòng, {df.shape[1]} cột")
    return df