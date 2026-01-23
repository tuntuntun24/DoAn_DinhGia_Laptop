# 💻 Đồ Án Nghiên Cứu: Dự Đoán Giá Laptop (XGBoost & Optuna)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Model-XGBoost_Optimized-orange)
![App](https://img.shields.io/badge/Web_App-Streamlit-red)

## 📖 Giới thiệu
Hệ thống định giá Laptop tự động sử dụng Machine Learning. Đồ án tập trung giải quyết bài toán dữ liệu nhỏ (~1300 mẫu) bằng thuật toán **XGBoost** kết hợp tối ưu hóa **Bayesian (Optuna)** để đạt độ chính xác cao và chống Overfitting.

## 📊 Dữ liệu & Phương pháp
* **Dữ liệu:** 1300 dòng, các đặc trưng: CPU, RAM, GPU, Màn hình, Trọng lượng...
* **Xử lý:** Làm sạch, chuẩn hóa đơn vị, One-Hot Encoding, Log Transformation cho biến giá (Price).

## 🔬 Thực nghiệm & Kết quả
Nhóm đã thử nghiệm 3 mô hình và áp dụng kỹ thuật **Fine-tuning** (tinh chỉnh) tham số chuyên sâu:

| Mô hình | R2 Score (Test) | MAE (Sai số) | Đánh giá |
| :--- | :--- | :--- | :--- |
| Linear Regression | 70.35% | ~2.11 tr VNĐ | Underfitting |
| Random Forest | 82.66% | ~1.63 tr VNĐ | Tốt |
| **XGBoost (Final)** | **85.07%** | **~1.51 tr VNĐ** | **Tốt nhất** |

> **Điểm nhấn:** Sử dụng **Optuna** để tìm bộ tham số tối ưu và **Early Stopping** để kiểm soát Overfitting (Chênh lệch Train/Test chỉ ~11%).

## 🚀 Cài đặt & Sử dụng
1. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Huấn luyện lại (Optional):**
   ```bash
   python 3_train_model.py
   ```
   *(File này đã được cập nhật bộ tham số tối ưu nhất, không cần chạy lại Optuna)*

3. **Chạy Web App:**
   ```bash
   streamlit run app.py
   ```

---
## 👨‍💻 Tác giả
* **Sinh viên:** [Điền Tên Bạn]
* **Trường:** Đại học Công nghiệp Hà Nội (HaUI)
* **Đồ án:** Nghiên cứu Khoa học / Tốt nghiệp