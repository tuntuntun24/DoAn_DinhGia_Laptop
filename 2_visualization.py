import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import master_pipeline  # Sử dụng lại logic chuẩn
import os

# Cấu hình giao diện chuẩn báo cáo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (22, 10)

# ==========================================
# CHẠY PHÂN TÍCH & VẼ BIỂU ĐỒ
# ==========================================
print("1. Đang xử lý dữ liệu...")
try:
    df_train = pd.read_csv('data/laptops_train.csv')
    df_test = pd.read_csv('data/laptops_test.csv')
except FileNotFoundError:
    print("⚠️ LỖI: Không tìm thấy file trong thư mục 'data/'")
    exit()

df_raw = pd.concat([df_train, df_test], ignore_index=True)

# DÙNG UTILS ĐỂ LÀM SẠCH (Code gọn hơn hẳn cũ)
df = master_pipeline(df_raw)

# Đổi đơn vị tiền tệ sang Triệu Đồng
df['Price'] = df['Price'] / 1_000_000
print("-> Đã chuyển đổi giá tiền sang đơn vị: Triệu VNĐ")

print("2. Đang vẽ biểu đồ gộp (HD)...")
fig, axes = plt.subplots(1, 2)

# --- HÌNH 1 (TRÁI): PHÂN BỐ GIÁ ---
sns.histplot(df['Price'], kde=True, color='#1f77b4', bins=30, ax=axes[0])
axes[0].set_title('PHÂN BỐ GIÁ LAPTOP THỰC TẾ', fontsize=18, fontweight='bold', pad=20)
axes[0].set_xlabel('Giá tiền (Triệu VNĐ)', fontsize=14)
axes[0].set_ylabel('Số lượng máy', fontsize=14)

# --- HÌNH 2 (PHẢI): HEATMAP TƯƠNG QUAN ---
# Lưu ý: 'Screen Size' đã bị drop trong master_pipeline để tính PPI,
# nên ta chỉ quan tâm tương quan giữa PPI và Giá (PPI quan trọng hơn Size)
cols_to_corr = ['Price', 'RAM', 'Weight', 'PPI']
sns.heatmap(df[cols_to_corr].corr(), annot=True, cmap='RdBu_r', fmt=".2f",
            linewidths=1, linecolor='white', annot_kws={"size": 14}, ax=axes[1])
axes[1].set_title('MỨC ĐỘ ẢNH HƯỞNG ĐẾN GIÁ (HEATMAP)', fontsize=18, fontweight='bold', pad=20)

plt.tight_layout(pad=3.0, w_pad=5.0)

# Lưu ảnh
if not os.path.exists('reports'):
    os.makedirs('reports')

file_name = 'reports/distribution_heatmap.png'
plt.savefig(file_name)
print(f"✅ ĐÃ XONG! Ảnh mới đã được lưu: {file_name}")
plt.show()