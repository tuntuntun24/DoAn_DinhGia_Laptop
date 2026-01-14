# 💻 HỆ THỐNG ĐỊNH GIÁ & CHIẾN LƯỢC KINH DOANH LAPTOP (AI POWERED)

> **Đồ án Thực tập Tốt nghiệp - Đại học Công Nghiệp Hà Nội (HaUI)**

## 📖 Giới thiệu (Overview)

Dự án này xây dựng một hệ thống **Machine Learning** khép kín, từ khâu thu thập, làm sạch dữ liệu đến việc huấn luyện mô hình AI để dự đoán giá Laptop. Đặc biệt, hệ thống tích hợp **Web App** hỗ trợ người kinh doanh tính toán biên lợi nhuận (Profit Margin) và đưa ra chiến lược nhập hàng thông minh.

## 🚀 Tính năng nổi bật
### 1. Phân tích dữ liệu & AI
* **Data Pipeline tự động:** Quy trình làm sạch và chuẩn hóa dữ liệu thông qua `master_pipeline` (xử lý RAM, SSD/HDD, Độ phân giải màn hình...).
* **Mô hình mạnh mẽ:** Sử dụng thuật toán **XGBoost Regressor** kết hợp kỹ thuật **GridSearchCV** để tối ưu hóa siêu tham số.
* **Độ chính xác cao:**
    * R² Score (Độ phù hợp): **~86%**
    * MAE (Sai số tuyệt đối): **~1.5 triệu VNĐ**

### 2. Ứng dụng Web (Streamlit)
* **Định giá theo cấu hình:** Nhập cấu hình (RAM, CPU, GPU, Màn hình...) và nhận giá thị trường gợi ý ngay lập tức.
* **Bài toán kinh doanh (Business Intelligence):**
    * Tính toán giá nhập hàng và lợi nhuận ròng trên từng máy.
    * Dự báo doanh thu tổng dựa trên số lượng bán mục tiêu.
    * **Cảnh báo rủi ro:** Hệ thống tự động cảnh báo nếu biên lợi nhuận quá mỏng (<10%) hoặc đánh giá tiềm năng nếu lợi nhuận cao (>25%).

## 🛠 Công nghệ sử dụng (Tech Stack)
| Lĩnh vực | Công nghệ / Thư viện |
| :--- | :--- |
| **Ngôn ngữ** | Python 3.9+ |
| **Xử lý dữ liệu** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Web Framework** | Streamlit |
| **Visualization** | Matplotlib, Seaborn |
| **Deploy** | Pickle (Serialization) |

## 📸 Demo Ứng dụng
*(Bạn hãy thay thế đường dẫn ảnh dưới đây bằng ảnh chụp màn hình thực tế từ dự án của bạn)*

### 1. Giao diện nhập thông số kỹ thuật
![Input Interface](https://via.placeholder.com/800x400?text=Giao+dien+nhap+lieu+Streamlit)

### 2. Kết quả định giá & Phân tích lợi nhuận
![Result Interface](https://via.placeholder.com/800x400?text=Ket+qua+dinh+gia+va+Loi+nhuan)

## 📂 Cấu trúc dự án
Dự án được tổ chức theo quy trình Data Science chuẩn:

```text
DoAn_DinhGia_Laptop/
├── data/                      # Chứa dữ liệu thô và test
│   ├── laptops_train.csv
│   └── laptops_test.csv
├── models/                    # Chứa mô hình đã huấn luyện
│   ├── laptop_price_model.pkl # Model XGBoost đã train
│   └── model_columns.pkl      # Danh sách cột đặc trưng
├── reports/                   # Báo cáo, biểu đồ phân tích
├── 1_data_analysis.py         # Phân tích khám phá dữ liệu (EDA)
├── 2_visualization.py         # Trực quan hóa (Heatmap, Distribution)
├── 3_train_model.py           # Huấn luyện, đánh giá & lưu Model
├── app.py                     # Source code Web App (Main)
├── utils.py                   # Các hàm tiện ích & Pipeline xử lý
├── requirements.txt           # Danh sách thư viện phụ thuộc
└── README.md                  # Tài liệu dự án
```

## ⚙️ Hướng dẫn cài đặt & Chạy
Yêu cầu hệ thống: Đã cài đặt **Python** và **Git**.

**Bước 1: Clone dự án**
```bash
git clone https://github.com/tuntuntun24/doan_dinhgia_laptop.git
cd doan_dinhgia_laptop
```

**Bước 2: Cài đặt thư viện**
```bash
pip install -r requirements.txt
```

**Bước 3: Huấn luyện mô hình (Tùy chọn)**
*Nếu bạn muốn training lại mô hình từ đầu để cập nhật dữ liệu mới:*
```bash
python 3_train_model.py
```

**Bước 4: Khởi chạy ứng dụng**
```bash
streamlit run app.py
```
*Trình duyệt sẽ tự động mở tại địa chỉ: `http://localhost:8501`*

## 👨‍💻 Tác giả
* **Sinh viên:** Chu Phú Thành
* **Trường:** Đại học Công Nghiệp Hà Nội (HaUI)
* **Học phần:** Đồ án tốt nghiệp
* **Liên hệ:** cpttt2004@gmail.com

---
**Disclaimer:** Dự án phục vụ mục đích học tập và nghiên cứu. Dữ liệu giá có thể thay đổi tùy thuộc vào thời điểm thị trường.
