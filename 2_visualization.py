import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker
from utils import master_pipeline  # Sử dụng hàm xử lý chuẩn của dự án

# Cấu hình giao diện chung
sns.set_style("whitegrid")

# Tạo thư mục lưu ảnh nếu chưa có
if not os.path.exists('reports'):
    os.makedirs('reports')

# ==========================================
# 1. ĐỌC VÀ XỬ LÝ DỮ LIỆU
# ==========================================
print("-> Đang tải và xử lý dữ liệu...")
try:
    df_train = pd.read_csv('data/laptops_train.csv')
    df_test = pd.read_csv('data/laptops_test.csv')
    df_raw = pd.concat([df_train, df_test], ignore_index=True)

    # --- [QUAN TRỌNG] CHUYỂN ĐỔI TIỀN TỆ (VNĐ) ---
    # Nhân tỷ giá 3.05 và hệ số thị trường 0.7
    df_raw['Price'] = df_raw['Price'] * 3.05 * 0.7

    # Chạy qua Pipeline để tạo các cột quan trọng như PPI, Weight (float)
    df = master_pipeline(df_raw)

except Exception as e:
    print(f"❌ Lỗi: {e}")
    exit()

# ==========================================
# 2. VẼ ẢNH 1: PHÂN BỐ GIÁ (HISTOGRAM)
# ==========================================
print("-> Đang vẽ biểu đồ 1: Phân bố giá...")
plt.figure(figsize=(12, 6))

# Vẽ Histogram
sns.histplot(df['Price'], kde=True, color='#1f77b4', bins=30)
plt.title('PHÂN BỐ GIÁ LAPTOP (THỊ TRƯỜNG VN)', fontsize=16, fontweight='bold')
plt.xlabel('Giá niêm yết (VNĐ)', fontsize=12)
plt.ylabel('Số lượng máy', fontsize=12)

# Định dạng trục X thành tiền Việt (VD: 20,000,000)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))

# Lưu ảnh 1
file_path1 = 'reports/price_distribution.png'
plt.savefig(file_path1, bbox_inches='tight')
print(f"✅ Đã lưu ảnh 1: {file_path1}")
plt.close()  # Đóng hình để giải phóng bộ nhớ

# ==========================================
# 3. VẼ ẢNH 2: MỨC ĐỘ ẢNH HƯỞNG (HEATMAP)
# ==========================================
print("-> Đang vẽ biểu đồ 2: Mức độ ảnh hưởng (Correlation)...")
plt.figure(figsize=(10, 8))

# Chọn các cột số quan trọng để xem tương quan
cols_to_analyze = ['Price', 'RAM', 'Weight', 'PPI', 'CPU_Freq']

# Tính ma trận tương quan
corr_matrix = df[cols_to_analyze].corr()

# Vẽ Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt=".2f",
            linewidths=1, linecolor='white', annot_kws={"size": 12})

plt.title('MỨC ĐỘ ẢNH HƯỞNG CÁC THÔNG SỐ ĐẾN GIÁ', fontsize=16, fontweight='bold')

# Lưu ảnh 2
file_path2 = 'reports/feature_correlation.png'
plt.savefig(file_path2, bbox_inches='tight')
print(f"✅ Đã lưu ảnh 2: {file_path2}")
plt.close()

print("🎉 HOÀN TẤT! Bạn hãy vào thư mục 'reports/' để lấy 2 ảnh mới nhé.")