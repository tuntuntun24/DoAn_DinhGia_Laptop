import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# ==========================================
# 1. CẤU HÌNH & IMPORT TIỆN ÍCH
# ==========================================
print("--- 🚀 KHỞI ĐỘNG HỆ THỐNG HUẤN LUYỆN (AI POWERED) ---")

# Import hàm xử lý dữ liệu từ file utils.py
try:
    from utils import master_pipeline
except ImportError:
    print("❌ LỖI CRITICAL: Không tìm thấy file 'utils.py' hoặc hàm 'master_pipeline'.")
    exit()

# Kiểm tra thư mục và file dữ liệu
if not os.path.exists('data/laptops_train.csv') or not os.path.exists('data/laptops_test.csv'):
    print("⚠️ LỖI: Không tìm thấy file csv trong thư mục 'data/'.")
    exit()

# ==========================================
# 2. CHUẨN BỊ DỮ LIỆU (DATA PREPARATION)
# ==========================================
print("-> 📂 Đang tải dữ liệu...")
df_train = pd.read_csv('data/laptops_train.csv')
df_test = pd.read_csv('data/laptops_test.csv')
df = pd.concat([df_train, df_test], ignore_index=True)

print("-> 🧹 Đang làm sạch dữ liệu (Data Cleaning via Pipeline)...")
df_clean = master_pipeline(df)

# Mã hóa One-Hot (One-Hot Encoding)
# Lưu ý: Việc này tạo ra các cột như 'Company_Dell', 'Company_Apple'...
df_encoded = pd.get_dummies(df_clean, columns=['Manufacturer', 'Category', 'CPU_Brand', 'GPU_Brand', 'OS'])

# Tách biến độc lập (X) và biến mục tiêu (y)
X = df_encoded.drop(columns=['Price'])
y = df_encoded['Price']

# Log Transform biến giá tiền (Giúp phân phối chuẩn hơn, mô hình học tốt hơn)
y_log = np.log(y)

# Chia tập dữ liệu: 85% Train - 15% Test
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.15, random_state=42)

print(f"-> Kích thước dữ liệu huấn luyện: {X_train.shape}")
print("-" * 40)

# ==========================================
# 3. HUẤN LUYỆN & SO SÁNH (TRAINING & EVALUATION)
# ==========================================

# Hàm tiện ích để in kết quả đánh giá
def evaluate_model(model, name, X_test, y_test_log):
    y_pred = np.exp(model.predict(X_test)) # Chuyển ngược từ Log về giá thực tế
    actual = np.exp(y_test_log)
    r2 = r2_score(actual, y_pred)
    mae = mean_absolute_error(actual, y_pred)
    print(f"🔹 {name:<20} | R2: {r2:.4f} | MAE: {mae:,.0f} VNĐ")
    return r2

# --- MODEL 1: LINEAR REGRESSION ---
lr = LinearRegression()
lr.fit(X_train, y_train_log)
evaluate_model(lr, "Linear Regression", X_test, y_test_log)

# --- MODEL 2: RANDOM FOREST ---
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train_log)
evaluate_model(rf, "Random Forest", X_test, y_test_log)

# ==========================================
# 4. XGBOOST NÂNG CAO (HYPERPARAMETER TUNING)
# ==========================================
print("\n-> ⏳ Đang chạy Grid Search tối ưu hóa XGBoost (AI Model)...")
print("   (Quá trình này có thể mất vài phút để tìm tham số tốt nhất)")

xgb_base = xgb.XGBRegressor(random_state=42)

# Lưới tham số "hạng nặng" để đạt độ chính xác cao
param_grid = {
    'n_estimators': [1000, 1500],       # Số lượng cây lớn để học sâu
    'learning_rate': [0.05],            # Tốc độ học chậm và chắc
    'max_depth': [6, 7],                # Độ sâu vừa đủ để bắt pattern phức tạp
    'subsample': [0.8],                 # Chỉ học 80% dữ liệu mỗi cây để tránh Overfitting
    'colsample_bytree': [0.8]           # (Mới) Chỉ dùng 80% số cột đặc trưng mỗi cây
}

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=3, verbose=1, n_jobs=-1, scoring='r2'
)

grid_search.fit(X_train, y_train_log)

best_xgb = grid_search.best_estimator_
print(f"✅ Tham số tối ưu: {grid_search.best_params_}")

# Đánh giá Model tốt nhất
r2_xgb = evaluate_model(best_xgb, "XGBoost (Tuned)", X_test, y_test_log)

# ==========================================
# 5. KIỂM TRA ĐỘ LỆCH (OVERFITTING CHECK)
# ==========================================
print("\n-> 🔍 Kiểm tra độ ổn định mô hình (Overfitting Check):")
y_pred_train = np.exp(best_xgb.predict(X_train))
r2_train = r2_score(np.exp(y_train_log), y_pred_train)

print(f"   + Độ chính xác trên tập TRAIN: {r2_train:.2%} (Lý thuyết)")
print(f"   + Độ chính xác trên tập TEST:  {r2_xgb:.2%}  (Thực tế)")

if r2_train - r2_xgb > 0.15:
    print("   ⚠️ CẢNH BÁO: Mô hình có dấu hiệu học vẹt (Overfitting).")
else:
    print("   ✅ ĐÁNH GIÁ: Mô hình học tốt, độ ổn định cao.")

print("-" * 40)

# ==========================================
# 6. LƯU MODEL (SAVING)
# ==========================================
if not os.path.exists('models'):
    os.makedirs('models')

print("💾 Đang lưu mô hình vào thư mục 'models/'...")

# 1. Lưu Model AI (XGBoost)
with open('models/laptop_price_model.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)

# 2. Lưu danh sách cột (Rất quan trọng cho Web App)
with open('models/model_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print(f"✅ HOÀN TẤT! Đã lưu model với độ chính xác R2 = {r2_xgb:.2%}")
print("   Sẵn sàng tích hợp vào Streamlit App.")