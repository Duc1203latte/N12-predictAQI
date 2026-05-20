import pandas as pd
import numpy as np

def clean_data(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df[['year','month','day','hour']]) #ghép thành 1 cột datetime chuẩn
    df = df.sort_values('datetime').reset_index(drop=True) #sắp xếp tăng dần theo thời gian
    df = df.drop(columns=['No','year','month','day','hour'])
    #Đổi các giá trị âm hoặc lớn hơn > 1000 thành NaN
    df['PM2.5'] = df['PM2.5'].where(df['PM2.5'] >= 0, np.nan)
    df['PM2.5'] = df['PM2.5'].where(df['PM2.5'] <= 1000, np.nan)
    df = df.set_index('datetime')
    df['PM2.5'] = df['PM2.5'].interpolate(method='linear', limit = 5) #chỉ fill tối đa 5 giờ liên tiếp bị thiếu
    numeric_cols = df.select_dtypes(include = 'number').columns #Nội suy các cột còn lại
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit = 3)
    df = df.reset_index()
    cbwd_dummies = pd.get_dummies(df['wd'], prefix='wind') #chuyển cột text cbwd thành các cột số 0/1,
    df = pd.concat([df, cbwd_dummies], axis=1)
    df = df.drop(columns=['wd'])
    # Chuyển True/False → 1/0
    wind_cols = [c for c in df.columns if c.startswith('wind_')]
    df[wind_cols] = df[wind_cols].astype(int)
    df = df.dropna(subset=['PM2.5'])
    print(f"cleaning xong: {df.shape[0]} dong, {df.shape[1]} cot")
    return df
