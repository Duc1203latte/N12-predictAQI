# PM2.5 Air Quality Prediction

## Cấu trúc project
pm25_forecast/
├─ data/raw/          ← tự download CSV từ Kaggle bỏ vào đây
├─ src/
│   ├─ cleaning.py
│   ├─ features.py
│   └─ splitting.py
├─ models/
├─ requirements.txt
└─ main.py

## Setup
pip install -r requirements.txt

## Dataset
Download tại: https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set
Dùng file: PRSA_Data_Dongsi_20130301-20170228.csv
Bỏ vào thư mục: data/raw/

## Đọc qua để hiểu cấu trúc
https://docs.google.com/document/d/1954pzx6vGOAPiVdPN6RT6uQFUpdy7Sk-HRAyU-xOnUc/edit?usp=sharing

## Chạy pipeline
python main.py