# 💻 Đồ Án Nghiên Cứu: Dự Đoán Giá Laptop (XGBoost & Optuna)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Model-XGBoost_Optimized-orange)
![App](https://img.shields.io/badge/Web_App-Streamlit-red)

## 📖 Giới thiệu
Hệ thống định giá Laptop tự động sử dụng Machine Learning. Đồ án tập trung giải quyết bài toán dữ liệu nhỏ (~1300 mẫu) bằng thuật toán **XGBoost** kết hợp tối ưu hóa **Bayesian (Optuna)** để đạt độ chính xác cao và chống Overfitting.

## 📊 Dữ liệu & Phương pháp
* **Dữ liệu:** 1300 dòng, bao gồm các đặc trưng: CPU, RAM, GPU, Màn hình, Trọng lượng...
* **Xử lý:** Làm sạch, chuẩn hóa đơn vị, One-Hot Encoding, Log Transformation.
* **Tiền tệ:** Dữ liệu được chuyển đổi từ INR sang **VNĐ** có hiệu chỉnh theo thị trường Việt Nam (Hệ số 0.7).

## 🔬 Thực nghiệm & Kết quả
Nhóm đã thử nghiệm 3 mô hình và áp dụng kỹ thuật **Fine-tuning** (tinh chỉnh) tham số chuyên sâu. Kết quả thực nghiệm mới nhất:

| Mô hình | R2 Score (Test) | MAE (Sai số TB) | Đánh giá |
| :--- | :--- | :--- | :--- |
| Linear Regression | 70.35% | ~4.51 tr VNĐ | Underfitting |
| Random Forest | 82.82% | ~3.44 tr VNĐ | Tốt |
| **XGBoost (Final)** | **85.13%** | **~3.23 tr VNĐ** | **Tốt nhất** |

> **Điểm nhấn:** Sử dụng **Optuna** tìm tham số tối ưu và **Early Stopping** kiểm soát Overfitting (Chênh lệch Train/Test ~11.9%).

## 🚀 Cài đặt & Sử dụng
1. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Huấn luyện lại (Optional):**
   ```bash
   python 3_train_model.py
   ```
   *(File đã cập nhật bộ tham số tối ưu và tỷ giá VNĐ)*

3. **Chạy Web App:**
   ```bash
   streamlit run app.py
   ```

---
## 👨‍💻 Tác giả
* **Sinh viên:** Chu Phú Thành
* **Trường:** Đại học Công nghiệp Hà Nội (HaUI)
* **Đồ án:** Nghiên cứu Khoa học