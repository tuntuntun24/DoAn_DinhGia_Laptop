import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


# ==========================================
# 1. HÀM DỌN DẸP DỮ LIỆU (Giống các bước trước)
# ==========================================
def clean_data(df):
    df = df.copy()
    # 1. Xử lý RAM, Weight, Screen Size
    df['RAM'] = df['RAM'].astype(str).str.replace('GB', '').astype(int)
    df['Weight'] = df['Weight'].astype(str).str.replace('kg', '').astype(float)
    df['Screen Size'] = pd.to_numeric(df['Screen Size'].astype(str).str.replace('"', ''), errors='coerce')

    # 2. Xử lý Màn hình & PPI
    df['Touchscreen'] = df['Screen'].apply(lambda x: 1 if 'Touchscreen' in str(x) else 0)
    df['IPS'] = df['Screen'].apply(lambda x: 1 if 'IPS' in str(x) else 0)

    df['X_res'] = df['Screen'].astype(str).str.extract(r'(\d+)x\d+').astype(int)
    df['Y_res'] = df['Screen'].astype(str).str.extract(r'\d+x(\d+)').astype(int)
    df['PPI'] = (((df['X_res'] ** 2) + (df['Y_res'] ** 2)) ** 0.5) / df['Screen Size']
    df.drop(columns=['X_res', 'Y_res', 'Screen Size', 'Screen'], inplace=True)

    # 3. Xử lý CPU
    df['CPU_Name'] = df['CPU'].apply(lambda x: " ".join(x.split()[0:3]))

    def fetch_processor(text):
        if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
            return text
        else:
            if text.split()[0] == 'Intel':
                return 'Other Intel Processor'
            else:
                return 'AMD Processor'

    df['CPU_Brand'] = df['CPU_Name'].apply(fetch_processor)
    df['CPU_Freq'] = df['CPU'].str.extract(r'(\d+(?:\.\d+)?)GHz').astype(float)
    df.drop(columns=['CPU', 'CPU_Name'], inplace=True)

    # 4. Xử lý Ổ cứng (SSD/HDD)
    def extract_storage(row):
        storage = str(row).upper()
        ssd = 0;
        hdd = 0
        parts = storage.split('+')
        for part in parts:
            capacity = 0
            match = re.search(r'(\d+)', part)
            if match:
                capacity = int(match.group(1))
                if 'TB' in part: capacity *= 1024
            if 'SSD' in part or 'FLASH' in part: ssd += capacity
            if 'HDD' in part: hdd += capacity
        return pd.Series([ssd, hdd])

    df[['SSD', 'HDD']] = df['Storage'].apply(extract_storage)
    df.drop(columns=['Storage'], inplace=True)

    # 5. Xử lý GPU & OS
    df['GPU_Brand'] = df['GPU'].apply(lambda x: x.split()[0])
    df['GPU_Brand'] = df['GPU_Brand'].apply(lambda x: x if x in ['Intel', 'Nvidia', 'AMD'] else 'Other')
    df.drop(columns=['GPU'], inplace=True)

    def cat_os(inp):
        if 'Windows' in inp:
            return 'Windows'
        elif 'Mac' in inp or 'mac' in inp:
            return 'Mac'
        else:
            return 'Others/No OS/Linux'

    df['OS'] = df['Operating System'].apply(cat_os)
    df.drop(columns=['Operating System', 'Operating System Version', 'Model Name'], inplace=True)  # Bỏ cột không dùng

    return df


# ==========================================
# 2. CHUẨN BỊ DỮ LIỆU
# ==========================================
print("--- BẮT ĐẦU HUẤN LUYỆN ---")
# Đọc và gộp dữ liệu
df_train = pd.read_csv('laptops_train.csv')
df_test = pd.read_csv('laptops_test.csv')
df = pd.concat([df_train, df_test], ignore_index=True)

# Làm sạch
df_clean = clean_data(df)

# Mã hóa One-Hot (Chuyển chữ thành số cho máy học)
# Lưu ý: Ta giữ lại các cột Category để One-Hot
df_encoded = pd.get_dummies(df_clean, columns=['Manufacturer', 'Category', 'CPU_Brand', 'GPU_Brand', 'OS'])

# Tách biến X (Thông số) và y (Giá tiền)
X = df_encoded.drop(columns=['Price'])
y = df_encoded['Price']

# Log Transform giá tiền (Để phân bố chuẩn hơn, mô hình học tốt hơn)
y_log = np.log(y)

# Chia tập Train (85%) và Test (15%)
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.15, random_state=42)

print(f"Dữ liệu huấn luyện: {X_train.shape[0]} dòng")
print(f"Dữ liệu kiểm tra:   {X_test.shape[0]} dòng")
print("-" * 30)

# ==========================================
# 3. SO SÁNH 3 MÔ HÌNH (RẤT QUAN TRỌNG VỚI ĐỒ ÁN)
# ==========================================

# --- MODEL 1: LINEAR REGRESSION (Cơ bản nhất) ---
lr = LinearRegression()
lr.fit(X_train, y_train_log)
y_pred_lr = np.exp(lr.predict(X_test))  # Phải exp ngược lại vì lúc đầu đã log
r2_lr = r2_score(np.exp(y_test_log), y_pred_lr)
mae_lr = mean_absolute_error(np.exp(y_test_log), y_pred_lr)
print(f"1. Linear Regression: R2 = {r2_lr:.4f} | MAE = {mae_lr:,.0f} VNĐ")

# --- MODEL 2: RANDOM FOREST (Mạnh hơn) ---
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train_log)
y_pred_rf = np.exp(rf.predict(X_test))
r2_rf = r2_score(np.exp(y_test_log), y_pred_rf)
mae_rf = mean_absolute_error(np.exp(y_test_log), y_pred_rf)
print(f"2. Random Forest:     R2 = {r2_rf:.4f} | MAE = {mae_rf:,.0f} VNĐ")

# --- MODEL 3: XGBOOST (Hiện đại nhất) ---
xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train_log)
y_pred_xgb = np.exp(xgb_model.predict(X_test))
r2_xgb = r2_score(np.exp(y_test_log), y_pred_xgb)
mae_xgb = mean_absolute_error(np.exp(y_test_log), y_pred_xgb)
print(f"3. XGBoost (AI):      R2 = {r2_xgb:.4f} | MAE = {mae_xgb:,.0f} VNĐ")

print("-" * 30)

# ==========================================
# 4. CHỌN MÔ HÌNH TỐT NHẤT VÀ LƯU
# ==========================================
# Chúng ta chọn XGBoost làm Final Model vì thường nó tốt nhất
if r2_xgb > 0.80:
    print("✅ ĐÁNH GIÁ: Mô hình XGBoost đạt độ chính xác TỐT (>80%).")
    print("-> Đang lưu mô hình vào file 'laptop_price_model.pkl'...")

    # Lưu Model
    with open('laptop_price_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

    # Lưu danh sách cột (Cực kỳ quan trọng để App chạy được)
    with open('model_columns.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

    print("-> Đã lưu thành công! Sẵn sàng chạy App.")
else:
    print("⚠️ CẢNH BÁO: Độ chính xác chưa cao. Cần xem lại dữ liệu.")