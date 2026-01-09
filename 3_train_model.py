import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# IMPORT HÀM XỬ LÝ TỪ FILE UTILS (MỚI)
from utils import master_pipeline

# ==========================================
# 1. CHUẨN BỊ DỮ LIỆU
# ==========================================
print("--- BẮT ĐẦU HUẤN LUYỆN ---")

# Đọc dữ liệu từ thư mục 'data/'
try:
    df_train = pd.read_csv('data/laptops_train.csv')
    df_test = pd.read_csv('data/laptops_test.csv')
except FileNotFoundError:
    print("⚠️ LỖI: Không tìm thấy file csv. Hãy kiểm tra lại thư mục 'data/'")
    exit()

# Gộp dữ liệu
df = pd.concat([df_train, df_test], ignore_index=True)

# SỬ DỤNG HÀM TỪ UTILS ĐỂ LÀM SẠCH (Thay vì viết lại code dài dòng)
print("-> Đang xử lý dữ liệu (Data Cleaning)...")
df_clean = master_pipeline(df)

# Mã hóa One-Hot (Giữ nguyên logic cũ)
df_encoded = pd.get_dummies(df_clean, columns=['Manufacturer', 'Category', 'CPU_Brand', 'GPU_Brand', 'OS'])

# Tách biến X (Thông số) và y (Giá tiền)
X = df_encoded.drop(columns=['Price'])
y = df_encoded['Price']

# Log Transform giá tiền
y_log = np.log(y)

# Chia tập Train (85%) và Test (15%)
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.15, random_state=42)

print(f"-> Dữ liệu huấn luyện: {X_train.shape[0]} dòng")
print(f"-> Dữ liệu kiểm tra:   {X_test.shape[0]} dòng")
print("-" * 30)

# ==========================================
# 2. HUẤN LUYỆN & SO SÁNH (Giữ nguyên logic cũ)
# ==========================================

# --- MODEL 1: LINEAR REGRESSION ---
lr = LinearRegression()
lr.fit(X_train, y_train_log)
y_pred_lr = np.exp(lr.predict(X_test))
r2_lr = r2_score(np.exp(y_test_log), y_pred_lr)
mae_lr = mean_absolute_error(np.exp(y_test_log), y_pred_lr)
print(f"1. Linear Regression: R2 = {r2_lr:.4f} | MAE = {mae_lr:,.0f} VNĐ")

# --- MODEL 2: RANDOM FOREST ---
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train_log)
y_pred_rf = np.exp(rf.predict(X_test))
r2_rf = r2_score(np.exp(y_test_log), y_pred_rf)
mae_rf = mean_absolute_error(np.exp(y_test_log), y_pred_rf)
print(f"2. Random Forest:     R2 = {r2_rf:.4f} | MAE = {mae_rf:,.0f} VNĐ")

# --- MODEL 3: XGBOOST ---
xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train_log)
y_pred_xgb = np.exp(xgb_model.predict(X_test))
r2_xgb = r2_score(np.exp(y_test_log), y_pred_xgb)
mae_xgb = mean_absolute_error(np.exp(y_test_log), y_pred_xgb)
print(f"3. XGBoost (AI):      R2 = {r2_xgb:.4f} | MAE = {mae_xgb:,.0f} VNĐ")

print("-" * 30)

# ==========================================
# 3. LƯU MODEL VÀO THƯ MỤC MODELS
# ==========================================
if r2_xgb > 0.80:
    print("✅ ĐÁNH GIÁ: Mô hình XGBoost tốt. Đang lưu...")

    # Lưu Model vào thư mục 'models/'
    with open('models/laptop_price_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

    # Lưu danh sách cột vào thư mục 'models/'
    with open('models/model_columns.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

    print("-> Đã lưu thành công vào thư mục 'models/'! Sẵn sàng chạy App.")
else:
    print("⚠️ CẢNH BÁO: Độ chính xác thấp dưới 80%.")