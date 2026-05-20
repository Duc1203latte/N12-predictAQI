from src.cleaning    import clean_data
from src.feature import build_features
from src.scaling_split import scale_and_split

if __name__ == "__main__":
    df = clean_data("data/raw/PRSA_Data_Dongsi_20130301-20170228.csv")
    df = build_features(df)
    #print(df.shape)
    # Xuất ra file CSV để xem
    #df.to_csv("data/processed/features_output1.csv", index=False)
    #print("Đã lưu vào data/processed/features_output1.csv")
    X_train, X_test, Y_train, Y_test = scale_and_split(df)
    print("X_train:", X_train.shape)
    print("X_test: ", X_test.shape)
    print("Y_train:", Y_train.shape)
    print("Y_test: ", Y_test.shape)