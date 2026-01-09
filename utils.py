import pandas as pd
import numpy as np
import re


def clean_ram_weight(df):
    """Xử lý cột RAM và Weight: Bỏ chữ, lấy số"""
    # Ép kiểu về string trước khi replace để tránh lỗi nếu dữ liệu đã là số
    df['RAM'] = df['RAM'].astype(str).str.replace('GB', '').astype(int)
    df['Weight'] = df['Weight'].astype(str).str.replace('kg', '').astype(float)
    return df


def calculate_ppi(df):
    """Tính toán chỉ số PPI từ độ phân giải và kích thước màn hình"""
    # Tách độ phân giải
    # Sử dụng regex để bắt chuỗi số trước và sau chữ 'x'
    df['X_res'] = df['Screen'].astype(str).str.extract(r'(\d+)x\d+').astype(int)
    df['Y_res'] = df['Screen'].astype(str).str.extract(r'\d+x(\d+)').astype(int)

    # Xử lý kích thước màn hình (bỏ dấu ngoặc kép nếu có)
    df['Screen Size'] = pd.to_numeric(df['Screen Size'].astype(str).str.replace('"', ''), errors='coerce')

    # Công thức tính PPI
    df['PPI'] = (((df['X_res'] ** 2) + (df['Y_res'] ** 2)) ** 0.5) / df['Screen Size']

    # Xóa các cột phụ sau khi tính xong để nhẹ dữ liệu
    df.drop(columns=['X_res', 'Y_res', 'Screen', 'Screen Size'], inplace=True, errors='ignore')
    return df


def process_cpu(df):
    """Phân loại CPU thành 5 nhóm chính"""
    # Lấy 3 từ đầu tiên của tên CPU
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
    # Lấy tốc độ CPU (GHz)
    df['CPU_Freq'] = df['CPU'].str.extract(r'(\d+(?:\.\d+)?)GHz').astype(float)

    # Xóa cột cũ
    df.drop(columns=['CPU', 'CPU_Name'], inplace=True)
    return df


def process_storage(df):
    """Tách dung lượng SSD và HDD từ cột Storage"""

    def extract_storage(row):
        storage = str(row).upper()
        ssd = 0
        hdd = 0
        parts = storage.split('+')
        for part in parts:
            capacity = 0
            # Tìm số trong chuỗi (ví dụ 128GB -> 128)
            match = re.search(r'(\d+)', part)
            if match:
                capacity = int(match.group(1))
                # Nếu là TB thì nhân với 1024
                if 'TB' in part: capacity *= 1024

            # Cộng dồn vào SSD hoặc HDD tùy loại
            if 'SSD' in part or 'FLASH' in part: ssd += capacity
            if 'HDD' in part: hdd += capacity
        return pd.Series([ssd, hdd])

    df[['SSD', 'HDD']] = df['Storage'].apply(extract_storage)
    df.drop(columns=['Storage'], inplace=True)
    return df


def process_gpu_os(df):
    """Xử lý cột GPU và Hệ điều hành"""
    # Xử lý GPU: Chỉ lấy hãng sản xuất (Intel, Nvidia, AMD)
    df['GPU_Brand'] = df['GPU'].apply(lambda x: x.split()[0])
    df['GPU_Brand'] = df['GPU_Brand'].apply(lambda x: x if x in ['Intel', 'Nvidia', 'AMD'] else 'Other')
    df.drop(columns=['GPU'], inplace=True)

    # Xử lý OS: Gom nhóm
    def cat_os(inp):
        if 'Windows' in inp:
            return 'Windows'
        elif 'Mac' in inp or 'mac' in inp:
            return 'Mac'
        else:
            return 'Others/No OS/Linux'

    df['OS'] = df['Operating System'].apply(cat_os)
    # Xóa các cột không cần thiết
    df.drop(columns=['Operating System', 'Operating System Version', 'Model Name'], inplace=True, errors='ignore')
    return df


def master_pipeline(df):
    """
    Hàm tổng hợp chạy toàn bộ quy trình làm sạch.
    Chỉ cần gọi hàm này là dữ liệu được xử lý từ A-Z.
    """
    df = df.copy()

    # 1. RAM & Weight
    df = clean_ram_weight(df)

    # 2. Xử lý màn hình (Touchscreen, IPS, PPI)
    # Lưu ý: Logic Touchscreen/IPS đơn giản nên để đây cũng được
    df['Touchscreen'] = df['Screen'].apply(lambda x: 1 if 'Touchscreen' in str(x) else 0)
    df['IPS'] = df['Screen'].apply(lambda x: 1 if 'IPS' in str(x) else 0)
    df = calculate_ppi(df)

    # 3. Xử lý phần cứng khác
    df = process_cpu(df)
    df = process_storage(df)
    df = process_gpu_os(df)

    return df