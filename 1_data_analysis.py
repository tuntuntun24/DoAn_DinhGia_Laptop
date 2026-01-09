import pandas as pd

# ==========================================
# FILE A: TỔNG QUAN DỮ LIỆU
# ==========================================

# 1. Đọc dữ liệu (Đã cập nhật đường dẫn vào thư mục data/)
print("--- 1. ĐANG ĐỌC DỮ LIỆU ---")
try:
    df_train = pd.read_csv('data/laptops_train.csv')
    df_test = pd.read_csv('data/laptops_test.csv')
except FileNotFoundError:
    print("⚠️ LỖI: Không tìm thấy file. Hãy kiểm tra lại thư mục 'data/'")
    exit()

# 2. Gộp Train và Test
df_train['Source'] = 'Train'
df_test['Source'] = 'Test'
df = pd.concat([df_train, df_test], ignore_index=True)

# 3. Báo cáo số liệu thống kê
print(f"\n--- 2. THỐNG KÊ SỐ LƯỢNG MẪU ---")
print(f"   - Số lượng tập Train: {len(df_train)} dòng")
print(f"   - Số lượng tập Test:  {len(df_test)} dòng")
print(f"   - TỔNG CỘNG:          {len(df)} dòng")

print("\n--- 3. KIỂM TRA DỮ LIỆU THIẾU (NULL) ---")
null_cols = df.isnull().sum()
print(null_cols[null_cols > 0])

print("\n--- 4. CẤU TRÚC DỮ LIỆU (INFO) ---")
print(df.info())

print("\n--- 5. MỘT VÀI DÒNG MẪU ---")
print(df.sample(5))