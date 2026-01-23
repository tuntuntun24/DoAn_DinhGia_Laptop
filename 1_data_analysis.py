import pandas as pd

# ==========================================
# FILE A: TỔNG QUAN DỮ LIỆU (Đã cập nhật tỷ giá VNĐ)
# ==========================================

print("--- 1. ĐANG ĐỌC DỮ LIỆU ---")
try:
    df_train = pd.read_csv('data/laptops_train.csv')
    df_test = pd.read_csv('data/laptops_test.csv')
except FileNotFoundError:
    print("⚠️ LỖI: Không tìm thấy file. Hãy kiểm tra lại thư mục 'data/'")
    exit()

# Gộp dữ liệu
df_train['Source'] = 'Train'
df_test['Source'] = 'Test'
df = pd.concat([df_train, df_test], ignore_index=True)

# --- [QUAN TRỌNG] CHUYỂN ĐỔI TIỀN TỆ ĐỂ THỐNG KÊ ĐÚNG ---
# Tỷ giá 3.05 và hệ số điều chỉnh thị trường 0.7
print("-> 💱 Đang cập nhật giá về thị trường Việt Nam...")
df['Price'] = df['Price'] * 3.05 * 0.7
# -------------------------------------------------------

print(f"\n--- 2. THỐNG KÊ SỐ LƯỢNG MẪU ---")
print(f"   - Tổng cộng: {len(df)} dòng")

print("\n--- 3. THỐNG KÊ GIÁ (VNĐ) ---")
# In ra để bạn kiểm tra xem giá có hợp lý không
print(df['Price'].describe().apply(lambda x: format(x, ',.0f')))

print("\n--- 4. KIỂM TRA DỮ LIỆU THIẾU ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n--- 5. CẤU TRÚC DỮ LIỆU ---")
print(df.info())