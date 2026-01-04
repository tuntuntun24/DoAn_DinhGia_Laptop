import pandas as pd
import numpy as np
import re
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# ==========================================
# 1. HÀM DỌN DẸP DỮ LIỆU (BẢN V3 - HOÀN THIỆN)
# ==========================================
def clean_data(df):
    df = df.copy()
    # 1. Cơ bản
    df['RAM'] = df['RAM'].astype(str).str.replace('GB', '').astype(int)
    df['Weight'] = df['Weight'].astype(str).str.replace('kg', '').astype(float)
    df['Screen Size'] = df['Screen Size'].astype(str).str.replace('"', '').astype(float)

    # 2. Màn hình & PPI
    df['Touchscreen'] = df['Screen'].apply(lambda x: 1 if 'Touchscreen' in str(x) else 0)
    df['IPS'] = df['Screen'].apply(lambda x: 1 if 'IPS' in str(x) else 0)

    df['X_res'] = df['Screen'].astype(str).str.extract(r'(\d+)x\d+').astype(int)
    df['Y_res'] = df['Screen'].astype(str).str.extract(r'\d+x(\d+)').astype(int)

    # PPI: Mật độ điểm ảnh (Rất quan trọng)
    df['PPI'] = (((df['X_res'] ** 2) + (df['Y_res'] ** 2)) ** 0.5) / df['Screen Size']

    df.drop(columns=['X_res', 'Y_res', 'Screen Size'], inplace=True)

    # 3. CPU
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

    # 4. Ổ cứng
    def extract_storage(row):
        storage = str(row).upper()
        ssd = 0
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

    # 5. GPU & OS
    df['GPU_Brand'] = df['GPU'].apply(lambda x: x.split()[0])
    df['GPU_Brand'] = df['GPU_Brand'].apply(lambda x: x if x in ['Intel', 'Nvidia', 'AMD'] else 'Other')

    def cat_os(inp):
        if 'Windows' in inp:
            return 'Windows'
        elif 'Mac' in inp or 'mac' in inp:
            return 'Mac'
        else:
            return 'Others/No OS/Linux'

    df['OS'] = df['Operating System'].apply(cat_os)

    # --- KHẮC PHỤC LỖI: KHÔNG XÓA CỘT CATEGORY NỮA ---
    # Chỉ xóa những cột thực sự không dùng
    cols_to_drop = ['Model Name', 'Screen', 'CPU', 'Storage', 'GPU',
                    'Operating System', 'Operating System Version', 'CPU_Name']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df


# ==========================================
# 2. CHẠY HUẤN LUYỆN
# ==========================================
print("1. Đang xử lý dữ liệu (V3 - Đã sửa lỗi thiếu cột Category)...")
df_train = pd.read_csv('laptops_train.csv')
df_test = pd.read_csv('laptops_test.csv')
df = pd.concat([df_train, df_test], ignore_index=True)

df_clean = clean_data(df)

# QUAN TRỌNG: Thêm 'Category' vào One-Hot Encoding
df_encoded = pd.get_dummies(df_clean, columns=['Manufacturer', 'Category', 'CPU_Brand', 'GPU_Brand', 'OS'])

df_model = df_encoded.dropna(subset=['Price'])
X = df_model.drop(columns=['Price'])
y = df_model['Price']

# Log Transform giá tiền
y_log = np.log(y)

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.15, random_state=42)

print("2. Đang huấn luyện AI...")
# Tinh chỉnh lại tham số XGBoost một chút cho tối ưu
model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train_log)

# Đánh giá
y_pred_log = model.predict(X_test)
y_pred = np.exp(y_pred_log)
y_test_real = np.exp(y_test_log)

r2 = r2_score(y_test_real, y_pred)
mae = mean_absolute_error(y_test_real, y_pred)

print("-" * 35)
print(f"🔥 KẾT QUẢ FINAL V3:")
print(f"   - Độ chính xác (R2 Score): {r2:.4f}")
print(f"   - Sai số trung bình (MAE): {mae:,.0f} VNĐ")
print("-" * 35)

if r2 > 0.85:
    print("🏆 TUYỆT VỜI! ĐÃ ĐẠT CHUẨN TỐT NGHIỆP.")
else:
    print("Vẫn chưa hài lòng? Chúng ta sẽ thử GridSearch (nhưng hơi lâu).")

# Lưu model
with open('laptop_price_model.pkl', 'wb') as f: pickle.dump(model, f)
with open('model_columns.pkl', 'wb') as f: pickle.dump(X.columns.tolist(), f)