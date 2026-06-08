DỰ BÁO CHẤT LƯỢNG KHÔNG KHÍ PM2.5 — NHÓM 12 

MÔ TẢ HỆ THỐNG
---------------
Hệ thống dự báo nồng độ bụi mịn PM2.5 tại 3 giờ tiếp theo
(t+1, t+2, t+3) sử dụng 3 mô hình học máy: Lasso Regression,
Ridge Regression và Random Forest. Dữ liệu từ bộ dataset
Beijing Multi-Site Air Quality (Kaggle), trạm Dongsi 2013-2017.

YÊU CẦU HỆ THỐNG
-----------------
- Python 3.9 trở lên
- pip (trình quản lý gói Python)

CÁC GÓI PHẦN MỀM SỬ DỤNG
--------------------------
pandas==2.x         Xử lý và phân tích dữ liệu dạng bảng
numpy==1.x          Tính toán số học
scikit-learn==1.x   Thuật toán học máy và đánh giá model
joblib==1.x         Lưu và load model đã train
matplotlib==3.x     Vẽ biểu đồ
streamlit==1.x      Xây dựng giao diện web

CÀI ĐẶT
--------
Bước 1: Clone hoặc giải nén source code

Bước 2: Tạo môi trường ảo (khuyến nghị)
  Windows:
    python -m venv .venv
    .venv\Scripts\activate

  Mac/Linux:
    python -m venv .venv
    source .venv/bin/activate

Bước 3: Cài đặt các gói phần mềm
    pip install -r requirements.txt

CHUẨN BỊ DỮ LIỆU
-----------------
Bước 1: Download dataset tại
    https://www.kaggle.com/datasets/sid321axn/
    beijing-multisite-airquality-data-set

Bước 2: Giải nén và bỏ toàn bộ 12 file CSV vào thư mục
    data/raw/

CHẠY CHƯƠNG TRÌNH
-----------------
Chạy theo đúng thứ tự sau: (hoặc chỉ cần chạy bước 4)

Bước 1: Tiền xử lý dữ liệu
    python main.py
    → Tạo ra file data/processed/features_output1.csv

Bước 2: Huấn luyện các mô hình
    python src/train_lasso.py
    python src/train_ridge.py
    python src/train_rf.py
    → Tạo ra các file .pkl trong thư mục models/

Bước 3: (Tùy chọn) Vẽ biểu đồ đánh giá
    python src/evaluate.py
    python src/train.py
    → Biểu đồ lưu tại outputs/charts/

Bước 4: Chạy giao diện web (đã pip install streamlit app)
    streamlit run app/app.py
    → Mở trình duyệt tại http://localhost:8501

HƯỚNG DẪN SỬ DỤNG GIAO DIỆN
-----------------------------
1. Upload file CSV của bất kỳ trạm nào vào ô Upload
2. Tab "Dự báo": kéo slider chọn thời điểm, xem kết quả
   dự báo PM2.5 tại t+1, t+2, t+3 kèm mức độ ô nhiễm
3. Tab "Đánh giá model": xem bảng so sánh RMSE/MAE/R²
   của 3 model
4. Tab "Biểu đồ": xem đồ thị Actual vs Predicted
   và Scatter plot

CẤU TRÚC THƯ MỤC
-----------------
predict_AQI/
├── app/app.py          Giao diện web Streamlit
├── data/raw/           Dữ liệu gốc CSV (tự download)
├── data/processed/     Dữ liệu sau tiền xử lý
├── models/             Model đã train (.pkl)
├── outputs/charts/     Biểu đồ xuất ra
├── src/                Mã nguồn chính
│   ├── cleaning.py     Làm sạch dữ liệu
│   ├── feature.py      Feature engineering
│   ├── scaling_split.py    Chuẩn hóa và chia tập
│   ├── train_lasso.py  Huấn luyện Lasso
│   ├── train_ridge.py  Huấn luyện Ridge
│   ├── train_rf.py     Huấn luyện Random Forest
│   ├── evaluate.py     Đánh giá và vẽ biểu đồ
│   └── train_curve.py  Learning curve
├── main.py             Pipeline tiền xử lý
├── requirements.txt    Danh sách thư viện
└── README.md           Hướng dẫn ngắn gọn

