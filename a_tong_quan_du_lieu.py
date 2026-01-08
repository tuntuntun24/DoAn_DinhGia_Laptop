import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Cấu hình giao diện chuẩn báo cáo
sns.set_style("whitegrid")
# Tăng kích thước khung hình lên (Rộng 22, Cao 10) để các chi tiết cực kỳ thoáng
plt.rcParams['figure.figsize'] = (22, 10)


# ==========================================
# 1. HÀM LÀM SẠCH DỮ LIỆU
# ==========================================
def clean_data_for_plot(df):
    df = df.copy()

    # Xử lý RAM, Weight (Bỏ chữ lấy số)
    df['RAM'] = df['RAM'].astype(str).str.replace('GB', '').astype(int)
    df['Weight'] = df['Weight'].astype(str).str.replace('kg', '').astype(float)

    # Xử lý PPI (Mật độ điểm ảnh)
    df['X_res'] = df['Screen'].astype(str).str.extract(r'(\d+)x\d+').astype(int)
    df['Y_res'] = df['Screen'].astype(str).str.extract(r'\d+x(\d+)').astype(int)
    df['Screen Size'] = pd.to_numeric(df['Screen Size'].astype(str).str.replace('"', ''), errors='coerce')
    df['PPI'] = (((df['X_res'] ** 2) + (df['Y_res'] ** 2)) ** 0.5) / df['Screen Size']

    return df


# ==========================================
# 2. CHẠY PHÂN TÍCH & VẼ BIỂU ĐỒ
# ==========================================
print("1. Đang xử lý dữ liệu...")
df_train = pd.read_csv('laptops_train.csv')
df_test = pd.read_csv('laptops_test.csv')
df_raw = pd.concat([df_train, df_test], ignore_index=True)

# Làm sạch dữ liệu
df = clean_data_for_plot(df_raw)

# --- QUAN TRỌNG: ĐỔI ĐƠN VỊ TIỀN TỆ SANG TRIỆU ĐỒNG ---
df['Price'] = df['Price'] / 1_000_000
print("-> Đã chuyển đổi giá tiền sang đơn vị: Triệu VNĐ")

print("2. Đang vẽ biểu đồ gộp (HD)...")

# Tạo khung chứa 2 biểu đồ
fig, axes = plt.subplots(1, 2)

# --- HÌNH 1 (TRÁI): PHÂN BỐ GIÁ (HISTOGRAM) ---
# kde=True: Vẽ thêm đường cong mật độ cho đẹp
sns.histplot(df['Price'], kde=True, color='#1f77b4', bins=30, ax=axes[0])
axes[0].set_title('PHÂN BỐ GIÁ LAPTOP THỰC TẾ', fontsize=18, fontweight='bold', pad=20)
axes[0].set_xlabel('Giá tiền (Triệu VNĐ)', fontsize=14)
axes[0].set_ylabel('Số lượng máy', fontsize=14)
axes[0].tick_params(axis='both', which='major', labelsize=12)

# --- HÌNH 2 (PHẢI): HEATMAP TƯƠNG QUAN ---
cols_to_corr = ['Price', 'RAM', 'Weight', 'PPI', 'Screen Size']
# Dùng bảng màu 'RdBu_r' (Red-Blue reverse) để màu đỏ thể hiện tương quan cao rõ hơn
sns.heatmap(df[cols_to_corr].corr(), annot=True, cmap='RdBu_r', fmt=".2f",
            linewidths=1, linecolor='white', annot_kws={"size": 14}, ax=axes[1])
axes[1].set_title('MỨC ĐỘ ẢNH HƯỞNG ĐẾN GIÁ (HEATMAP)', fontsize=18, fontweight='bold', pad=20)
axes[1].tick_params(axis='both', which='major', labelsize=12)

# --- CĂN CHỈNH KHOẢNG CÁCH (PADDING) ---
# w_pad=5.0: Đẩy 2 biểu đồ ra xa nhau hơn
# pad=3.0: Đẩy lề xung quanh ra
plt.tight_layout(pad=3.0, w_pad=5.0)

# Lưu ảnh
file_name = 'phan_tich_bieu_do.png'
plt.savefig(file_name)
print(f"✅ ĐÃ XONG! Ảnh mới đã được lưu: {file_name}")
plt.show()