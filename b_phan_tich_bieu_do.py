import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Cấu hình hiển thị cho đẹp
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")


# ==========================================
# 1. HÀM DỌN DẸP DỮ LIỆU (Tái sử dụng từ code của bạn)
# ==========================================
def clean_data(df):
    df = df.copy()
    # Xử lý cơ bản
    df['RAM'] = df['RAM'].astype(str).str.replace('GB', '').astype(int)
    df['Weight'] = df['Weight'].astype(str).str.replace('kg', '').astype(float)

    # Xử lý màn hình
    df['Touchscreen'] = df['Screen'].apply(lambda x: 1 if 'Touchscreen' in str(x) else 0)
    df['IPS'] = df['Screen'].apply(lambda x: 1 if 'IPS' in str(x) else 0)

    # PPI
    df['X_res'] = df['Screen'].astype(str).str.extract(r'(\d+)x\d+').astype(int)
    df['Y_res'] = df['Screen'].astype(str).str.extract(r'\d+x(\d+)').astype(int)
    # df['Screen Size'] = df['Screen Size'].astype(str).str.replace('"', '').astype(float) # Cẩn thận dòng này nếu dữ liệu có lỗi
    # Để an toàn hơn, thay dòng trên bằng:
    df['Screen Size'] = pd.to_numeric(df['Screen Size'].astype(str).str.replace('"', ''), errors='coerce')

    df['PPI'] = (((df['X_res'] ** 2) + (df['Y_res'] ** 2)) ** 0.5) / df['Screen Size']

    # CPU
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

    # Bộ nhớ
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

    # GPU & OS
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

    return df


# ==========================================
# 2. GỘP DỮ LIỆU & LÀM SẠCH
# ==========================================
print("--- ĐANG TẢI DỮ LIỆU ---")
df_train = pd.read_csv('laptops_train.csv')
df_test = pd.read_csv('laptops_test.csv')

# Đánh dấu nguồn gốc (để sau này biết dòng nào là train, dòng nào là test nếu cần)
df_train['Source'] = 'Train'
df_test['Source'] = 'Test'

# Gộp (Concat)
df_total = pd.concat([df_train, df_test], ignore_index=True)
print(f"Tổng số dòng sau khi gộp: {df_total.shape[0]} dòng")

print("\n--- ĐANG LÀM SẠCH DỮ LIỆU ---")
df_clean = clean_data(df_total)

# ==========================================
# 3. PHÂN TÍCH TỔNG QUAN (EDA)
# ==========================================

# 3.1 Xem phân bố giá (Target Variable)
plt.figure(figsize=(10, 5))
sns.histplot(df_clean['Price'], kde=True, color='blue')
plt.title('PHÂN BỐ GIÁ LAPTOP (VNĐ)')
plt.xlabel('Giá')
plt.ylabel('Số lượng máy')
plt.show()

# 3.2 Tương quan giữa các thông số và Giá
# Chỉ lấy các cột số để tính tương quan
numeric_cols = ['Price', 'RAM', 'Weight', 'PPI', 'SSD', 'HDD', 'CPU_Freq']
plt.figure(figsize=(10, 8))
sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('BIỂU ĐỒ TƯƠNG QUAN (HEATMAP)')
plt.show()

# 3.3 Giá trung bình theo Hãng
plt.figure(figsize=(12, 6))
avg_price_brand = df_clean.groupby('Manufacturer')['Price'].mean().sort_values(ascending=False)
sns.barplot(x=avg_price_brand.index, y=avg_price_brand.values, palette='viridis')
plt.xticks(rotation=45)
plt.title('GIÁ TRUNG BÌNH THEO THƯƠNG HIỆU')
plt.ylabel('Giá trung bình')
plt.show()

print("\n--- HOÀN TẤT ---")
print(df_clean.info())