import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================================
# 1. CẤU HÌNH TRANG WEB
# ============================================
st.set_page_config(
    page_title="Hệ Thống Định Giá & Chiến Lược Laptop",
    page_icon="💻",
    layout="wide"
)

# CSS làm đẹp giao diện
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
        st.error("⚠️ LỖI: Không tìm thấy file mô hình. Hãy chạy file '2_huan_luyen_mo_hinh.py' trước!")
        return None, None


model, model_columns = load_data()

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
        # --- CẬP NHẬT: Màn hình chọn theo list có sẵn ---
        screen_size = st.selectbox("Màn hình (Inch)", [11.6, 12.0, 12.5, 13.3, 14.0, 15.6, 17.3])
    with col_s2:
        # --- CẬP NHẬT: Cân nặng chọn theo list phổ biến ---
        weight = st.selectbox("Nặng (kg)", [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 4.0])
        touchscreen = st.selectbox("Cảm ứng", ["Không", "Có"])

    ips = st.selectbox("Tấm nền IPS", ["Không", "Có"])
    resolution = st.selectbox("Độ phân giải",
                              ['1366x768', '1920x1080', '2560x1440', '3840x2160', '2880x1800', '2560x1600',
                               '2304x1440'])

    st.markdown("---")
    cpu_brand = st.selectbox("CPU", ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Other Intel Processor',
                                     'AMD Processor'])

    # --- CẬP NHẬT: CPU GHz chọn theo list có sẵn ---
    cpu_freq = st.selectbox("Tốc độ CPU (GHz)",
                            [0.9, 1.1, 1.2, 1.3, 1.6, 1.8, 2.0, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.2, 3.6])

    ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1000, 2000])
    hdd = st.selectbox("HDD (GB)", [0, 500, 1000, 2000])

    gpu_brand = st.selectbox("Card đồ họa (GPU)", ['Intel', 'Nvidia', 'AMD', 'Other'])
    os = st.selectbox("Hệ điều hành", ['Windows', 'Mac', 'Others/No OS/Linux'])

    st.write("")
    btn_predict = st.button("🚀 ĐỊNH GIÁ & PHÂN TÍCH", type="primary")

# ============================================
# 4. XỬ LÝ DỰ ĐOÁN & HIỂN THỊ (PHẦN CHÍNH)
# ============================================
st.markdown('<div class="main-header">HỆ THỐNG GỢI Ý CHIẾN LƯỢC GIÁ (AI POWERED)</div>', unsafe_allow_html=True)

if btn_predict and model:
    # --- A. CHUẨN BỊ DỮ LIỆU ĐẦU VÀO ---
    # 1. Tính PPI (Mật độ điểm ảnh)
    try:
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    except:
        ppi = 100  # Giá trị mặc định nếu lỗi

    # 2. Tạo bảng dữ liệu rỗng đúng chuẩn model
    input_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

    # 3. Điền các giá trị số
    input_data['RAM'] = ram
    input_data['Weight'] = weight
    input_data['PPI'] = ppi
    input_data['CPU_Freq'] = cpu_freq
    input_data['SSD'] = ssd
    input_data['HDD'] = hdd
    input_data['Touchscreen'] = 1 if touchscreen == "Có" else 0
    input_data['IPS'] = 1 if ips == "Có" else 0


    # 4. Điền các cột One-Hot
    def set_one_hot(col_prefix, value):
        col_name = f"{col_prefix}_{value}"
        if col_name in input_data.columns:
            input_data[col_name] = 1


    set_one_hot('Manufacturer', brand)
    set_one_hot('Category', category)
    set_one_hot('CPU_Brand', cpu_brand)
    set_one_hot('GPU_Brand', gpu_brand)
    set_one_hot('OS', os)

    # --- B. DỰ ĐOÁN ---
    # Model trả về log(giá), cần np.exp để ra giá thật
    predicted_log = model.predict(input_data)
    predicted_price = np.exp(predicted_log)[0]

    # Lưu session state
    st.session_state['price'] = predicted_price
    st.session_state['has_run'] = True

# --- C. HIỂN THỊ KẾT QUẢ ---
if st.session_state.get('has_run'):
    price = st.session_state['price']

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown(f"""
        <div class="price-card">
            <h3 style="margin-top:0;">🏷️ GIÁ KHUYẾN NGHỊ</h3>
            <h1 style="color: #D32F2F; font-size: 48px; margin: 10px 0;">{price:,.0f} VNĐ</h1>
            <p><i>Độ tin cậy: ~86%</i></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # st.markdown('<div class="strategy-card">', unsafe_allow_html=True)
        st.subheader("📈 BÀI TOÁN LỢI NHUẬN (Profit Strategy)")

        # Giả định mặc định: Giá vốn = 75% giá bán
        default_cost = int(price * 0.75)

        c1, c2 = st.columns(2)
        with c1:
            input_cost = st.number_input("Giá nhập hàng (VNĐ)", value=default_cost, step=100000, format="%d")
        with c2:
            target_qty = st.number_input("Số lượng dự kiến bán (Tháng)", value=10, step=1)

        profit_per_unit = price - input_cost
        margin = (profit_per_unit / price) * 100 if price > 0 else 0
        total_profit = profit_per_unit * target_qty

        st.write("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Lợi nhuận/Máy", f"{profit_per_unit:,.0f} đ", delta=f"{margin:.1f}% Margin")
        m2.metric("Doanh thu dự kiến", f"{price * target_qty:,.0f} đ")
        m3.metric("Tổng lợi nhuận Ròng", f"{total_profit:,.0f} đ", delta_color="normal")

        if margin < 10:
            st.warning("⚠️ Cảnh báo: Biên lợi nhuận mỏng (<10%). Cần tối ưu chi phí nhập!")
        elif margin > 25:
            st.success("✅ Tuyệt vời: Sản phẩm có biên lợi nhuận cao (>25%).")
        else:
            st.info("ℹ️ Ổn định: Biên lợi nhuận ở mức tiêu chuẩn (10-25%).")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👈 Hãy chọn cấu hình laptop ở thanh bên trái và bấm nút 'ĐỊNH GIÁ & PHÂN TÍCH' để bắt đầu.")