# 💻 HỆ THỐNG ĐỊNH GIÁ & CHIẾN LƯỢC KINH DOANH LAPTOP (AI POWERED)

Đồ án tốt nghiệp xây dựng mô hình Machine Learning dự đoán giá Laptop và ứng dụng Web App hỗ trợ ra quyết định kinh doanh.

## 📌 Giới thiệu

Dự án giải quyết bài toán khó khăn trong việc định giá laptop trên thị trường cũ/mới. Hệ thống sử dụng thuật toán **XGBoost** để học từ dữ liệu cấu hình phần cứng và đưa ra mức giá gợi ý với độ chính xác cao.

- **Độ chính xác (R2 Score):** ~86%
- **Sai số trung bình (MAE):** ~1.5 triệu VNĐ
- **Công nghệ:** Python, Scikit-learn, XGBoost, Streamlit.

## 🚀 Hướng dẫn cài đặt & Chạy (Quick Start)

**Bước 1: Tải dự án về máy**

    git clone https://github.com/tuntuntun24/doan_dinhgia_laptop.git
    cd doan_dinhgia_laptop

**Bước 2: Cài đặt các thư viện cần thiết**

    pip install -r requirements.txt

**Bước 3: Khởi động Ứng dụng Web**

    streamlit run d_chay_ung_dung.py

*(Sau khi chạy lệnh này, trình duyệt sẽ tự động mở trang web định giá)*

## 📂 Cấu trúc dự án

Để thuận tiện cho việc theo dõi luồng xử lý dữ liệu, code được chia thành 4 phần chính:

1. **`a_tong_quan_du_lieu.py`**
   - Đọc và kiểm tra dữ liệu thô.
   - Thống kê số lượng mẫu (Train/Test).

2. **`b_phan_tich_bieu_do.py`**
   - Làm sạch dữ liệu phục vụ trực quan hóa.
   - Vẽ biểu đồ Phân bố giá và Biểu đồ nhiệt (Heatmap) để phân tích tương quan.

3. **`c_huan_luyen_mo_hinh.py`**
   - Xử lý đặc trưng (Feature Engineering).
   - Huấn luyện và so sánh 3 thuật toán: Linear Regression, Random Forest, XGBoost.
   - Lưu model tốt nhất (`.pkl`).

4. **`d_chay_ung_dung.py`**
   - Giao diện Web App (Streamlit).
   - Tích hợp bài toán tính toán lợi nhuận và tư vấn chiến lược giá.

## 👨‍💻 Tác giả

- **Sinh viên:** Chu Phú Thành
- **Lớp/Trường:** Đại học Công Nghiệp Hà Nội (HaUI)
- **Đồ án môn:**  Thực tập tốt nghiệp