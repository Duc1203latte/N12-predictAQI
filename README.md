Dự báo PM2.5 tại t+1, t+2, t+3 sử dụng dataset Beijing Multisite (trạm Dongsi).

⚙️ Cài đặt
bashpip install -r requirements.txt
📥 Dataset
Download tại Kaggle

🚀 Chạy theo thứ tự
bash# 1. Preprocessing
python main.py

# 2. Train 3 model
python src/train_lasso.py
python src/train_ridge.py
python src/train_rf.py

# 3. Chạy app để test
streamlit run app/app.py (mở terminal, gõ lênh này. nhớ pip install streamlit trước)

🧪 Test nhiều trạm
Vào app → upload CSV của bất kỳ trạm nào → xem metrics + biểu đồ của cả 3 model.
Ghi kết quả vào bảng báo cáo, đặc biệt chú ý:

R² có giảm nhiều so với Dongsi không?
RF có tốt hơn Linear ở t+2/t+3 không?