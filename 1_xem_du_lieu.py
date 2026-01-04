import pandas as pd

# 1. Đọc file dữ liệu
print("Đang đọc dữ liệu...")
df = pd.read_csv('laptops_train.csv')

# 2. In ra 5 dòng đầu tiên xem nó trông thế nào
print("\n--- 5 DÒNG ĐẦU TIÊN ---")
print(df.head())

# 3. Xem tổng quan (có bao nhiêu dòng, cột nào là số, cột nào là chữ)
print("\n--- THÔNG TIN DỮ LIỆU ---")
print(df.info())