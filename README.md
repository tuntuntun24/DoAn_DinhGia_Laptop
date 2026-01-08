# 💻 Laptop Price Prediction & Business Strategy System

Hệ thống dự đoán giá Laptop và tư vấn chiến lược kinh doanh sử dụng Machine Learning (XGBoost), được xây dựng bằng Python và Streamlit.

## 📌 Tổng quan
Đồ án này giải quyết bài toán định giá laptop cũ/mới trên thị trường, giúp người bán tối ưu lợi nhuận và người mua tránh bị hớ.
- **Độ chính xác mô hình:** R2 Score ~ 86%.
- **Sai số trung bình:** ~1.5 triệu VNĐ.
- **Công nghệ:** Python, Scikit-learn, XGBoost, Streamlit.

## 📂 Cấu trúc dự án
1. **`a_tong_quan_du_lieu.py`**: Kiểm tra, thống kê dữ liệu thô.
2. **`b_phan_tich_bieu_do.py`**: Trực quan hóa dữ liệu (EDA) & xuất biểu đồ báo cáo.
3. **`c_huan_luyen_mo_hinh.py`**: Huấn luyện AI, so sánh 3 mô hình & lưu model tốt nhất.
4. **`d_chay_ung_dung.py`**: Chạy Website Demo sản phẩm.

## 🚀 Hướng dẫn cài đặt
1. Clone dự án:
   ```bash
   git clone [https://github.com/tuntuntun24/doan_dinhgia_laptop.git](https://github.com/tuntuntun24/doan_dinhgia_laptop.git)
   
2. Cài đặt các thư viện cần thiết:
    ```bash
   pip install -r requirements.txt
   
3. Khởi động Ứng dụng Web:
   ```bash
   streamlit run d_chay_ung_dung.py
