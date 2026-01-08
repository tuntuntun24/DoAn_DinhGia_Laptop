import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================================
# 1. CẤU HÌNH TRANG WEB (GIAO DIỆN CŨ)
# ============================================
st.set_page_config(
    page_title="Hệ Thống Định Giá & Chiến Lược Laptop",
    page_icon="💻",
    layout="wide"
)

# CSS GIAO DIỆN GỐC (ĐÃ KHÔI PHỤC)
st.markdown("""
<style>
    .main-header {
        font-size: 32px; 
        font-weight: bold; 
        color: #1565C0; 
        text-align: center;
        margin-bottom: 25px;
        text-transform: uppercase;
    }
    .price-card {
        background-color: #E3F2FD;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #2196F3;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .strategy-card {
        background-color: #F1F8E9;
        padding: 25px;
        border-radius: 12px;
        border: 2px solid #66BB6A;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
        height: 50px;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# 2. TẢI MÔ HÌNH
# ============================================
@st.cache_resource
def load_data():
    try:
        with open('laptop_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model_columns.pkl', 'rb') as f:
            cols = pickle.load(f)
        return model, cols
    except:
        st.error("⚠️ LỖI: Không tìm thấy file mô hình. Hãy chạy file 'c_huan_luyen_mo_hinh.py' trước!")
        return None, None


model, model_columns = load_data()

# --- KHỞI TẠO SESSION STATE (ĐỂ SỬA LỖI LOAD LẠI) ---
if 'price' not in st.session_state:
    st.session_state['price'] = None

# ============================================
# 3. GIAO DIỆN NHẬP LIỆU (SIDEBAR TRÁI)
# ============================================
with st.sidebar:
    st.header("⚙️ THÔNG SỐ KỸ THUẬT")
    st.markdown("---")

    # Nhập liệu
    brand = st.selectbox("Thương hiệu",
                         ['Dell', 'Lenovo', 'HP', 'Asus', 'Acer', 'Apple', 'MSI', 'Toshiba', 'Samsung', 'Razer',
                          'Mediacom', 'Microsoft', 'Xiaomi', 'Vero', 'Chuwi', 'Google', 'Fujitsu', 'LG', 'Huawei'])
    category = st.selectbox("Loại máy",
                            ['Notebook', 'Ultrabook', 'Gaming', '2 in 1 Convertible', 'Workstation', 'Netbook'])

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
        screen_size = st.number_input("Màn hình (Inch)", min_value=10.0, max_value=18.0, value=15.6, step=0.1)
    with col_s2:
        weight = st.number_input("Nặng (kg)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
        touchscreen = st.selectbox("Cảm ứng", ["Không", "Có"])

    ips = st.selectbox("Tấm nền IPS", ["Không", "Có"])
    resolution = st.selectbox("Độ phân giải",
                              ['1366x768', '1920x1080', '2560x1440', '3840x2160', '2880x1800', '2560x1600',
                               '2304x1440'])

    st.markdown("---")
    cpu_brand = st.selectbox("CPU", ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Other Intel Processor',
                                     'AMD Processor'])
    cpu_freq = st.number_input("Tốc độ CPU (GHz)", min_value=0.5, max_value=5.0, value=2.5, step=0.1)

    ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1000, 2000])
    hdd = st.selectbox("HDD (GB)", [0, 500, 1000, 2000])

    gpu_brand = st.selectbox("Card đồ họa (GPU)", ['Intel', 'Nvidia', 'AMD', 'Other'])
    os = st.selectbox("Hệ điều hành", ['Windows', 'Mac', 'Others/No OS/Linux'])

    st.write("")
    btn_predict = st.button("🚀 ĐỊNH GIÁ & PHÂN TÍCH", type="primary")

# ============================================
# 4. XỬ LÝ DỰ ĐOÁN & HIỂN THỊ
# ============================================
st.markdown('<div class="main-header">HỆ THỐNG GỢI Ý CHIẾN LƯỢC GIÁ (AI POWERED)</div>', unsafe_allow_html=True)

# KHI BẤM NÚT -> TÍNH TOÁN VÀ LƯU VÀO SESSION STATE
if btn_predict and model:
    # 1. Tính PPI
    try:
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    except:
        ppi = 100

    # 2. Tạo input
    input_data = pd.DataFrame(index=[0], columns=model_columns)
    input_data = input_data.fillna(0)

    # 3. Điền giá trị
    input_data['RAM'] = ram
    input_data['Weight'] = weight
    input_data['PPI'] = ppi
    input_data['CPU_Freq'] = cpu_freq
    input_data['SSD'] = ssd
    input_data['HDD'] = hdd
    input_data['Touchscreen'] = 1 if touchscreen == "Có" else 0
    input_data['IPS'] = 1 if ips == "Có" else 0


    # 4. One-Hot Encoding
    def set_one_hot(col_prefix, value):
        col_name = f"{col_prefix}_{value}"
        if col_name in input_data.columns:
            input_data[col_name] = 1


    set_one_hot('Manufacturer', brand)
    set_one_hot('Category', category)
    set_one_hot('CPU_Brand', cpu_brand)
    set_one_hot('GPU_Brand', gpu_brand)
    set_one_hot('OS', os)

    # 5. Dự đoán & LƯU VÀO SESSION
    predicted_log = model.predict(input_data)
    predicted_price = np.exp(predicted_log)[0]

    st.session_state['price'] = predicted_price

# --- HIỂN THỊ KẾT QUẢ (DÙNG LAYOUT CŨ) ---
if st.session_state['price'] is not None:
    price = st.session_state['price']

    col1, col2 = st.columns([1, 1.5])

    with col1:
        # Giao diện Price Card cũ
        st.markdown(f"""
        <div class="price-card">
            <h3 style="margin-top:0; color: #1565C0;">🏷️ GIÁ KHUYẾN NGHỊ</h3>
            <h1 style="color: #D32F2F; font-size: 48px; margin: 10px 0;">{price:,.0f} VNĐ</h1>
            <p><i>Độ tin cậy của AI: ~86%</i></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Giao diện Strategy Card cũ
        st.markdown('<div class="strategy-card">', unsafe_allow_html=True)
        st.subheader("📈 BÀI TOÁN LỢI NHUẬN (Profit Strategy)")

        # Logic tính toán cũ
        default_cost = int(price * 0.75)

        c1, c2 = st.columns(2)
        with c1:
            input_cost = st.number_input("Giá nhập hàng (VNĐ)", value=default_cost, step=100000, format="%d")
        with c2:
            target_qty = st.number_input("Số lượng bán (Tháng)", value=10, step=1)

        profit_per_unit = price - input_cost
        margin = (profit_per_unit / price) * 100 if price > 0 else 0
        total_profit = profit_per_unit * target_qty

        st.write("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Lợi nhuận/Máy", f"{profit_per_unit:,.0f} đ", delta=f"{margin:.1f}% Margin")
        m2.metric("Doanh thu dự kiến", f"{price * target_qty:,.0f} đ")
        m3.metric("Tổng lãi ròng", f"{total_profit:,.0f} đ")

        if margin < 10:
            st.warning("⚠️ Cảnh báo: Biên lợi nhuận mỏng (<10%). Cần tối ưu chi phí nhập!")
        elif margin > 25:
            st.success("✅ Tuyệt vời: Sản phẩm có biên lợi nhuận cao (>25%).")
        else:
            st.info("ℹ️ Ổn định: Biên lợi nhuận ở mức tiêu chuẩn (10-25%).")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👈 Hãy chọn cấu hình laptop ở thanh bên trái và bấm nút 'ĐỊNH GIÁ & PHÂN TÍCH' để bắt đầu.")