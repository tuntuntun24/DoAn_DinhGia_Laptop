import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================================
# 1. CẤU HÌNH & CSS (Tách riêng cho gọn)
# ============================================
st.set_page_config(
    page_title="Hệ Thống Định Giá & Chiến Lược Laptop",
    page_icon="💻",
    layout="wide"
)


def local_css():
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


local_css()


# ============================================
# 2. TẢI MÔ HÌNH (Đã cập nhật đường dẫn models/)
# ============================================
@st.cache_resource
def load_data():
    try:
        # Đọc file từ thư mục 'models/'
        with open('models/laptop_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/model_columns.pkl', 'rb') as f:
            cols = pickle.load(f)
        return model, cols
    except FileNotFoundError:
        st.error("⚠️ LỖI: Không tìm thấy file mô hình. Hãy chạy file '3_train_model.py' trước!")
        return None, None


model, model_columns = load_data()

# Khởi tạo session state
if 'price' not in st.session_state:
    st.session_state['price'] = None

# ============================================
# 3. GIAO DIỆN NHẬP LIỆU (SIDEBAR - ĐÃ SẮP XẾP CHUẨN)
# ============================================
with st.sidebar:
    st.header("⚙️ CẤU HÌNH CHI TIẾT")
    st.markdown("---")

    # --- NHÓM 1: THƯƠNG HIỆU & PHÂN KHÚC ---
    st.subheader("🏷️ Nhận diện")
    brand = st.selectbox("Thương hiệu",
                         ['Acer', 'Apple', 'Asus', 'Chuwi', 'Dell', 'Fujitsu', 'Google', 'HP', 'Huawei', 'LG', 'Lenovo',
                          'MSI', 'Mediacom', 'Microsoft', 'Razer', 'Samsung', 'Toshiba', 'Vero', 'Xiaomi'])

    category = st.selectbox("Dòng sản phẩm",
                            ['2 in 1 Convertible', 'Gaming', 'Netbook', 'Notebook', 'Ultrabook', 'Workstation'],
                            index=3)  # Mặc định chọn Notebook

    st.markdown("---")

    # --- NHÓM 2: SỨC MẠNH XỬ LÝ ---
    st.subheader("🚀 Sức mạnh xử lý")
    cpu_brand = st.selectbox("Dòng CPU",
                             ['AMD Processor', 'Intel Core i3', 'Intel Core i5', 'Intel Core i7',
                              'Other Intel Processor'], index=2)  # Mặc định i5
    cpu_freq = st.number_input("Tốc độ CPU (GHz)", min_value=0.5, max_value=5.0, value=2.5, step=0.1)

    col_perf1, col_perf2 = st.columns(2)
    with col_perf1:
        # Đã sắp xếp từ bé đến lớn, index=3 tương ứng với mặc định 8GB
        ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
    with col_perf2:
        gpu_brand = st.selectbox("Card đồ họa", ['AMD', 'Intel', 'Nvidia', 'Other'])

    st.markdown("---")

    # --- NHÓM 3: TRẢI NGHIỆM HIỂN THỊ ---
    st.subheader("🖥️ Màn hình")

    # Danh sách Preset cũng được sắp xếp theo kích thước từ bé đến lớn
    screen_presets = {
        "13.3\" Full HD (1920x1080)": (13.3, "1920x1080"),
        "13.3\" Retina/QHD (2560x1600)": (13.3, "2560x1600"),
        "14.0\" Full HD (1920x1080)": (14.0, "1920x1080"),
        "15.6\" Full HD (1920x1080)": (15.6, "1920x1080"),
        "15.6\" 4K Ultra HD (3840x2160)": (15.6, "3840x2160"),
        "17.3\" Full HD (1920x1080)": (17.3, "1920x1080"),
    }

    selected_preset = st.selectbox("Chọn loại màn hình", list(screen_presets.keys()), index=3)
    preset_size, preset_res = screen_presets[selected_preset]

    if selected_preset == "Tùy chỉnh thông số...":
        col_scr1, col_scr2 = st.columns(2)
        with col_scr1:
            screen_size = st.number_input("Kích thước (Inch)", min_value=10.0, max_value=18.0, value=15.6, step=0.1)
        with col_scr2:
            # Sắp xếp độ phân giải từ bé đến lớn
            resolution = st.selectbox("Độ phân giải",
                                      ['1366x768', '1920x1080', '2304x1440', '2560x1440', '2560x1600', '2880x1800',
                                       '3840x2160'], index=1)
    else:
        screen_size = preset_size
        resolution = preset_res
        st.caption(f"Đang sử dụng: {screen_size} inch | {resolution}")

    col_panel1, col_panel2 = st.columns(2)
    with col_panel1:
        ips = st.selectbox("Tấm nền IPS", ["Không", "Có"])
    with col_panel2:
        touchscreen = st.selectbox("Cảm ứng", ["Không", "Có"])

    st.markdown("---")

    # --- NHÓM 4: LƯU TRỮ & DI ĐỘNG ---
    st.subheader("💾 Lưu trữ & Di động")
    col_st1, col_st2 = st.columns(2)
    with col_st1:
        # Sắp xếp từ bé đến lớn, index=2 tương ứng mặc định 256GB
        ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1000, 2000], index=2)
    with col_st2:
        hdd = st.selectbox("HDD (GB)", [0, 500, 1000, 2000])

    weight = st.number_input("Trọng lượng máy (kg)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    os = st.selectbox("Hệ điều hành", ['Mac', 'Others/No OS/Linux', 'Windows'], index=2)  # Mặc định Windows

    st.write("")
    btn_predict = st.button("🚀 ĐỊNH GIÁ & PHÂN TÍCH", type="primary")

# ============================================
# 4. XỬ LÝ DỰ ĐOÁN
# ============================================
st.markdown('<div class="main-header">HỆ THỐNG GỢI Ý CHIẾN LƯỢC GIÁ (AI POWERED)</div>', unsafe_allow_html=True)

if btn_predict and model:
    # 1. Tính toán PPI (Logic giống hệt Utils nhưng áp dụng cho đơn giá trị)
    try:
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    except:
        ppi = 100  # Giá trị mặc định nếu lỗi

    # 2. Tạo DataFrame chứa dữ liệu đầu vào (Toàn số 0 ban đầu)
    input_data = pd.DataFrame(index=[0], columns=model_columns)
    input_data = input_data.fillna(0)

    # 3. Điền các giá trị số (Numerical)
    input_data['RAM'] = ram
    input_data['Weight'] = weight
    input_data['PPI'] = ppi
    input_data['CPU_Freq'] = cpu_freq
    input_data['SSD'] = ssd
    input_data['HDD'] = hdd
    input_data['Touchscreen'] = 1 if touchscreen == "Có" else 0
    input_data['IPS'] = 1 if ips == "Có" else 0


    # 4. Điền các giá trị phân loại (Categorical - One Hot Encoding)
    # Hàm này tìm cột đúng tên (ví dụ: 'Manufacturer_Dell') và đánh dấu là 1
    def set_one_hot(col_prefix, value):
        col_name = f"{col_prefix}_{value}"
        if col_name in input_data.columns:
            input_data[col_name] = 1


    set_one_hot('Manufacturer', brand)
    set_one_hot('Category', category)
    set_one_hot('CPU_Brand', cpu_brand)
    set_one_hot('GPU_Brand', gpu_brand)
    set_one_hot('OS', os)

    # 5. Dự đoán
    try:
        predicted_log = model.predict(input_data)
        predicted_price = np.exp(predicted_log)[0]  # Exp ngược lại vì lúc train đã log
        st.session_state['price'] = predicted_price
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")

# ============================================
# 5. HIỂN THỊ KẾT QUẢ
# ============================================
if st.session_state['price'] is not None:
    price = st.session_state['price']

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown(f"""
        <div class="price-card">
            <h3 style="margin-top:0; color: #1565C0;">🏷️ GIÁ KHUYẾN NGHỊ</h3>
            <h1 style="color: #D32F2F; font-size: 48px; margin: 10px 0;">{price:,.0f} VNĐ</h1>
            <p><i>Độ tin cậy của AI: ~86%</i></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # st.markdown('<div class="strategy-card">', unsafe_allow_html=True)
        st.subheader("📈 BÀI TOÁN LỢI NHUẬN")

        # Mặc định giá nhập bằng 75% giá bán
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
            st.warning("⚠️ Biên lợi nhuận mỏng (<10%). Cần tối ưu chi phí nhập!")
        elif margin > 25:
            st.success("✅ Sản phẩm có biên lợi nhuận cao (>25%). Rất tiềm năng!")
        else:
            st.info("ℹ️ Biên lợi nhuận ở mức tiêu chuẩn (10-25%).")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👈 Hãy chọn cấu hình laptop ở thanh bên trái và bấm nút 'ĐỊNH GIÁ & PHÂN TÍCH' để bắt đầu.")