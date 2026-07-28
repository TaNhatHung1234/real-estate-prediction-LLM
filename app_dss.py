import streamlit as st
import pandas as pd
import json
import os
import google.generativeai as genai
from ml_engine import predict_house_price

# ==========================================
# 1. XÂY DỰNG RAG ENGINE (TRUY XUẤT TỪ CSV)
# ==========================================
@st.cache_data
def load_and_prep_rag_db():
    df = pd.read_csv('vietnam_housing_dataset.csv')
    df['Tỉnh thành'] = df['Tỉnh thành'].astype(str).str.strip().str.rstrip('.')
    return df

def get_rag_context(user_house, df_rag, top_k=3):
    """Truy xuất các căn hộ tương đồng nhất trong dataset làm dữ liệu RAG"""
    matched = df_rag[
        (df_rag['Tỉnh thành'] == user_house['Tỉnh thành']) & 
        (df_rag['Quận huyện'] == user_house['Quận huyện'])
    ]
    if len(matched) == 0:
        matched = df_rag[df_rag['Tỉnh thành'] == user_house['Tỉnh thành']]
        
    # Tính khoảng cách diện tích để tìm căn tương đồng
    matched = matched.copy()
    matched['diff_area'] = (matched['Diện tích'] - user_house['Diện tích']).abs()
    similar_houses = matched.sort_values(by='diff_area').head(top_k)
    
    rag_snippets = []
    for idx, row in similar_houses.iterrows():
        snippet = (
            f"Lịch sử giao dịch/BĐS tham chiếu tại {row['Quận huyện']}, {row['Tỉnh thành']}: "
            f"Diện tích {row['Diện tích']}m2, {row['Số tầng']} tầng, Pháp lý '{row['Pháp lý']}', "
            f"Nội thất '{row['Nội thất']}', Giá niêm yết/bán: {row['Giá bán']} tỷ VNĐ."
        )
        rag_snippets.append(snippet)
    return rag_snippets

# ==========================================
# 2. GIAO DIỆN STREAMLIT DSS
# ==========================================
st.set_page_config(page_title="DSS Hỗ Trợ Dự Đoán Giá BĐS Cho Người Bán", layout="wide")

st.title("🏡 Hệ Hỗ Trợ Quyết Định (DSS) Định Giá & Tư Vấn Bán BĐS")
st.markdown("---")

col_left, col_right = st.columns([1, 1.5])

# --- CỘT TRÁI: NHẬP THÔNG TIN BĐS ---
with col_left:
    st.subheader("📋 Nhập thông tin BĐS của bạn")
    
    df_db = load_and_prep_rag_db()
    provinces = sorted(df_db['Tỉnh thành'].dropna().unique().tolist())
    
    selected_province = st.selectbox("Tỉnh / Thành phố", provinces, index=provinces.index("Hồ Chí Minh") if "Hồ Chí Minh" in provinces else 0)
    
    districts = sorted(df_db[df_db['Tỉnh thành'] == selected_province]['Quận huyện'].dropna().unique().tolist())
    selected_district = st.selectbox("Quận / Huyện", districts)
    
    area = st.number_input("Diện tích (m²)", min_value=10.0, max_value=1000.0, value=85.0)
    frontage = st.number_input("Mặt tiền (m)", min_value=0.0, max_value=50.0, value=5.0)
    access_road = st.number_input("Đường vào (m)", min_value=0.0, max_value=50.0, value=6.0)
    floors = st.number_input("Số tầng", min_value=1.0, max_value=20.0, value=3.0)
    bedrooms = st.number_input("Số phòng ngủ", min_value=1.0, max_value=10.0, value=3.0)
    bathrooms = st.number_input("Số toilet", min_value=1.0, max_value=10.0, value=3.0)
    legal = st.selectbox("Pháp lý", ["Có sổ", "Hợp đồng mua bán", "Chưa rõ"])
    furniture = st.selectbox("Nội thất", ["Đầy đủ", "Cơ bản", "Không xác định"])
    direction = st.selectbox("Hướng nhà", ["Đông - Nam", "Đông - Bắc", "Tây - Nam", "Tây - Bắc", "Nam", "Bắc", "Đông", "Tây"])

    btn_predict = st.button("🚀 Chạy Mô Hình Định Giá & Khởi Tạo DSS", use_container_width=True)

# Khởi tạo Session State
if "dss_context" not in st.session_state:
    st.session_state.dss_context = None
if "messages" not in st.session_state:
    st.session_state.messages = []

if btn_predict:
    user_house = {
        'Tỉnh thành': selected_province,
        'Quận huyện': selected_district,
        'Diện tích': area,
        'Mặt tiền': frontage,
        'Đường vào': access_road,
        'Số tầng': floors,
        'Số phòng ngủ': bedrooms,
        'Số toilet': bathrooms,
        'Pháp lý': legal,
        'Nội thất': furniture,
        'Hướng nhà': direction
    }
    
    # 1. Chạy ML Engine
    ml_res = predict_house_price(user_house)
    
    # 2. Truy xuất RAG Engine
    rag_res = get_rag_context(user_house, df_db)
    
    # 3. Tổng hợp DSS Context
    st.session_state.dss_context = {
        "ml_outputs": ml_res,
        "rag_context": rag_res
    }
    st.session_state.messages = [] # Reset chat
    st.success("Đã hoàn tất tính toán định giá ML và truy xuất dữ liệu!")

# --- CỘT PHẢI: HIỂN THỊ KẾT QUẢ ML & CHATBOT DSS ---
with col_right:
    if st.session_state.dss_context is not None:
        ctx = st.session_state.dss_context
        ml_data = ctx["ml_outputs"]["ket_qua_dinh_gia_ml"]
        
        st.subheader("📊 Kết quả Phân tích Định lượng (ML Engine)")
        st.metric("Giá dự đoán Trung vị", f"{ml_data['gia_du_doan_trung_vi_ty']} Tỷ VNĐ")
        
        col1, col2, col3 = st.columns(3)
        col1.info(f"**Bán nhanh (1-2 tuần):**\n{ml_data['kich_ban_gia_de_xuat']['ban_nhanh_1_2_tuan']}")
        col2.success(f"**Giá thị trường (1 tháng):**\n{ml_data['kich_ban_gia_de_xuat']['gia_thi_truong_1_thang']}")
        col3.warning(f"**Kỳ vọng cao (2-3 tháng):**\n{ml_data['kich_ban_gia_de_xuat']['ky_vong_cao_2_3_thang']}")
        
        # Cấu hình API LLM (GEMINI)
        st.markdown("---")
        st.subheader("🔑 Cấu hình Gemini")
        api_key_input = st.text_input("Nhập Gemini API Key", type="password")

        if api_key_input:
            genai.configure(api_key=api_key_input)
            st.success("Đã kết nối API Key thành công!")
        else:
            st.warning("Vui lòng nhập API Key để dùng Chatbot.")


            
        st.markdown("---")
        st.subheader("💬 Cố vấn BĐS AI (LLM + RAG + Guardrails)")
        
        # Hiển thị lịch sử Chat
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        # Input Chat
        user_query = st.chat_input("Hỏi Cố vấn BĐS (Ví dụ: Tại sao giá nhà tôi lại ở mức này? Có nên sửa sang không?)...")
        
        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)
            
            # Tạo System Prompt với Guardrails
            system_prompt = f"""
Bạn là Chuyên gia Cố vấn Bất động sản cao cấp trong Hệ thống Hỗ trợ Quyết định (DSS) cho NGƯỜI BÁN.

==================================================
[STRICT GUARDRAILS - QUY TẮC NGUYÊN TẮC]
1. CHỈ SỬ DỤNG DỮ LIỆU ĐƯỢC CUNG CẤP TRONG KHỐI [CONTEXT DATA].
2. KHÔNG TỰ BỊA RA các con số, lịch sử giao dịch, tên chủ đầu tư, hoặc phí dịch vụ KHÔNG CÓ TRONG DATA.
3. Nếu người dùng hỏi thông tin không có trong CONTEXT DATA, hãy từ chối lịch sự: "Dữ liệu hiện tại chưa ghi nhận thông tin này. Tôi chỉ tư vấn dựa trên dữ liệu định giá ML và giao dịch RAG đã xác thực."
4. Tôn trọng 100% con số giá do ML định giá.
==================================================

[CONTEXT DATA]
1. Dữ liệu định giá ML & Nhà người bán:
{json.dumps(ctx['ml_outputs'], ensure_ascii=False, indent=2)}

2. Dữ liệu tham chiếu thị trường (RAG Retrieval):
{json.dumps(ctx['rag_context'], ensure_ascii=False, indent=2)}

[USER QUESTION]: {user_query}
"""
            with st.chat_message("assistant"):
                try:
                    model_gemini = genai.GenerativeModel('gemini-3.6-flash', generation_config={"temperature": 0.1})
                    response = model_gemini.generate_content(system_prompt)
                    bot_reply = response.text
                except Exception as e:
                    bot_reply = f"Lỗi kết nối LLM API: {str(e)}. Hãy kiểm tra API Key."
                
                st.write(bot_reply)
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    else:
        st.info("👈 Hãy chọn thông tin BĐS bên cột trái và bấm 'Chạy Mô Hình Định Giá' để khởi tạo DSS.")