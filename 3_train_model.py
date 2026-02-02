import pandas as pd
import numpy as np
import pickle
import os
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
# ⚠️ True: Chạy tìm kiếm tham số (Lâu). False: Dùng tham số có sẵn (Nhanh).
ENABLE_OPTUNA = False


# ==========================================
# 0. CÁC HÀM TIỆN ÍCH (DEFINITIONS)
# ==========================================
def evaluate_model(model, name, X_test, y_test_log):
    """Hàm đánh giá và in kết quả model"""
    y_pred = np.exp(model.predict(X_test))
    actual = np.exp(y_test_log)
    r2 = r2_score(actual, y_pred)
    mae = mean_absolute_error(actual, y_pred)
    print(f"🔹 {name:<20} | R2: {r2:.4f} | MAE: {mae:,.0f} VNĐ")
    return r2


def plot_feature_importance(model, feature_names, output_dir='reports'):
    """Chỉ vẽ biểu đồ và lưu file ảnh (Fix lỗi warning seaborn)"""
    print("\n📊 Đang vẽ biểu đồ Feature Importance...")

    if not hasattr(model, 'feature_importances_'):
        print("⚠️ Cảnh báo: Model không hỗ trợ 'feature_importances_'.")
        return

    # 1. Tạo DataFrame và sắp xếp
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Lấy Top 15 để vẽ cho đẹp
    feature_df = feature_df.sort_values(by='Importance', ascending=False).head(15)

    # 2. Vẽ biểu đồ (ĐÃ SỬA LỖI WARNING TẠI ĐÂY)
    plt.figure(figsize=(12, 8))

    # Thêm hue='Feature' và legend=False theo yêu cầu mới của seaborn
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_df, palette='mako', legend=False)

    plt.title('TOP ĐẶC TRƯNG ẢNH HƯỞNG ĐẾN GIÁ LAPTOP', fontsize=15, fontweight='bold')
    plt.xlabel('Mức độ đóng góp (Importance Score)', fontsize=12)
    plt.ylabel('Tên đặc trưng', fontsize=12)
    plt.tight_layout()

    # 3. Tạo thư mục và lưu ảnh
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_path = os.path.join(output_dir, 'feature_importance_xgboost.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ vào: '{img_path}'")
    plt.close()


# ==========================================
# 1. KHỞI ĐỘNG & LOAD DỮ LIỆU
# ==========================================
print("--- 🚀 KHỞI ĐỘNG HỆ THỐNG HUẤN LUYỆN (AI POWERED) ---")

try:
    from utils import master_pipeline
except ImportError:
    print("❌ LỖI: Không tìm thấy file 'utils.py'.")
    exit()

if not os.path.exists('data/laptops_train.csv'):
    print("⚠️ LỖI: Không tìm thấy file dữ liệu.")
    exit()

print("-> 📂 Đang tải dữ liệu...")
df_train = pd.read_csv('data/laptops_train.csv')
df_test = pd.read_csv('data/laptops_test.csv')
df = pd.concat([df_train, df_test], ignore_index=True)

# Chuyển đổi tiền tệ
print("-> 💱 Đang xử lý tiền tệ...")
df['Price'] = df['Price'] * 3.05 * 0.7

print("-> 🧹 Đang làm sạch dữ liệu...")
df_clean = master_pipeline(df)

# One-hot encoding
df_encoded = pd.get_dummies(df_clean, columns=['Manufacturer', 'Category', 'CPU_Brand', 'GPU_Brand', 'OS'])

X = df_encoded.drop(columns=['Price'])
y = df_encoded['Price']
y_log = np.log(y)

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.15, random_state=42)
print(f"-> Kích thước tập Train: {X_train.shape}")
print("-" * 40)

# ==========================================
# 2. HUẤN LUYỆN CƠ BẢN (BASELINE)
# ==========================================
lr = LinearRegression()
lr.fit(X_train, y_train_log)
evaluate_model(lr, "Linear Regression", X_test, y_test_log)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train_log)
evaluate_model(rf, "Random Forest", X_test, y_test_log)

# ==========================================
# 3. TỐI ƯU HÓA HYPERPARAMETER (OPTUNA)
# ==========================================
print("\n-> ⏳ Bắt đầu cấu hình XGBoost...")

if ENABLE_OPTUNA:
    print("   ⚠️ CHẾ ĐỘ TỐI ƯU ĐANG BẬT (Mất nhiều thời gian)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)


    def objective(trial):
        params = {
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': 0
        }
        model = xgb.XGBRegressor(**params)
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train_log, cv=cv, scoring='r2')
        return scores.mean()


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    print(f"   ✅ Tìm thấy tham số tốt nhất: R2 = {study.best_value:.4f}")
    best_params = study.best_params
    best_params.update({'n_estimators': 1000, 'n_jobs': -1, 'random_state': 42})
else:
    print("   ⏩ Bỏ qua Optuna (Dùng tham số đã lưu).")
    best_params = {
        'learning_rate': 0.06113883171486565,
        'max_depth': 5,
        'subsample': 0.7079585175427282,
        'colsample_bytree': 0.7714315790179074,
        'reg_alpha': 0.4917950397223208,
        'reg_lambda': 2.02634753580506,
        'n_estimators': 1000,
        'n_jobs': -1,
        'random_state': 42
    }

# ==========================================
# 4. HUẤN LUYỆN MODEL CUỐI CÙNG
# ==========================================
print("\n-> 🚀 Huấn luyện model XGBoost Final...")

final_model = xgb.XGBRegressor(**best_params, early_stopping_rounds=100)
final_model.fit(
    X_train, y_train_log,
    eval_set=[(X_test, y_test_log)],
    verbose=False
)

evaluate_model(final_model, "XGBoost (Optuna)", X_test, y_test_log)

# ==========================================
# 5. KIỂM TRA OVERFITTING
# ==========================================
print("\n-> 🔍 Kiểm tra độ ổn định:")
r2_train = r2_score(np.exp(y_train_log), np.exp(final_model.predict(X_train)))
r2_test = r2_score(np.exp(y_test_log), np.exp(final_model.predict(X_test)))

print(f"   + R2 Train: {r2_train:.2%}")
print(f"   + R2 Test:  {r2_test:.2%}")

if (r2_train - r2_test) > 0.15:
    print("   ⚠️ CẢNH BÁO: Có dấu hiệu Overfitting.")
else:
    print("   ✅ ĐÁNH GIÁ: Model học ổn định.")

# ==========================================
# 6. LƯU TRỮ VÀ XUẤT BÁO CÁO
# ==========================================
try:
    plot_feature_importance(final_model, X.columns)
except Exception as e:
    print(f"❌ Lỗi vẽ biểu đồ: {e}")

if not os.path.exists('models'):
    os.makedirs('models')

print("\n💾 Đang lưu model...")
with open('models/laptop_price_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

with open('models/model_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ HOÀN TẤT TOÀN BỘ QUÁ TRÌNH!")