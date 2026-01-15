import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

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
# 4. XGBOOST NÂNG CAO VỚI OPTUNA + EARLY STOPPING
# ==========================================
print("\n-> ⏳ Đang khởi động Optuna để tìm tham số tối ưu (AI Mode)...")


# --- A. ĐỊNH NGHĨA HÀM MỤC TIÊU (OBJECTIVE FUNCTION) ---
def objective(trial):
    params = {
        'n_estimators': 1000,
        # Cho phép học nhanh hơn một chút
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),

        # Cho phép cây sâu hơn một chút để bắt được các mẫu khó
        'max_depth': trial.suggest_int('max_depth', 5, 10),

        # Giữ nguyên để chống học vẹt
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),

        # GIẢM HÌNH PHẠT: Cho phép model linh hoạt hơn
        # Trước đây cho tới 10.0, giờ chỉ cho tối đa 2.0 hoặc 3.0
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 3.0),

        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0
    }

    cv_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]

        model = xgb.XGBRegressor(**params, early_stopping_rounds=100)  # Tăng kiên nhẫn lên 100

        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )

        preds = model.predict(X_val_fold)
        score = r2_score(y_val_fold, preds)
        cv_scores.append(score)

    return np.mean(cv_scores)

# --- B. CHẠY TỐI ƯU HÓA ---
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"\n✅ Đã tìm thấy tham số tốt nhất:")
print(f"   -> R2 trung bình (Cross-Validation): {study.best_value:.4f}")
print(f"   -> Bộ tham số: {study.best_params}")

# ==========================================
# 5. HUẤN LUYỆN LẠI MODEL CUỐI CÙNG (FINAL TRAINING)
# ==========================================
print("\n-> 🚀 Đang huấn luyện lại model tốt nhất trên toàn bộ tập Train...")

best_params = study.best_params
best_params['n_estimators'] = 1000
best_params['n_jobs'] = -1
best_params['random_state'] = 42

# === SỬA LỖI TẠI ĐÂY (Bước 2) ===
# Đưa early_stopping_rounds vào constructor của model cuối cùng
final_model = xgb.XGBRegressor(**best_params, early_stopping_rounds=100)

# Xóa early_stopping_rounds khỏi hàm fit
final_model.fit(
    X_train, y_train_log,
    eval_set=[(X_test, y_test_log)],
    verbose=False
)

evaluate_model(final_model, "XGBoost (Optuna)", X_test, y_test_log)

# ==========================================
# 6. KIỂM TRA OVERFITTING & LƯU MODEL
# ==========================================
print("\n-> 🔍 Kiểm tra độ ổn định mô hình (Overfitting Check):")
y_pred_train = np.exp(final_model.predict(X_train))
r2_train = r2_score(np.exp(y_train_log), y_pred_train)

y_pred_test_final = np.exp(final_model.predict(X_test))
r2_test_final = r2_score(np.exp(y_test_log), y_pred_test_final)

print(f"   + Độ chính xác trên tập TRAIN: {r2_train:.2%}")
print(f"   + Độ chính xác trên tập TEST:  {r2_test_final:.2%}")

diff = r2_train - r2_test_final
if diff > 0.15:
    print(f"   ⚠️ CẢNH BÁO: Chênh lệch {diff:.2%} -> Vẫn còn dấu hiệu Overfitting nhẹ.")
else:
    print(f"   ✅ TUYỆT VỜI: Chênh lệch {diff:.2%} -> Model học rất ổn định!")

if not os.path.exists('models'):
    os.makedirs('models')

print("\n💾 Đang lưu mô hình vào thư mục 'models/'...")
with open('models/laptop_price_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

with open('models/model_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ HOÀN TẤT TOÀN BỘ QUÁ TRÌNH!")