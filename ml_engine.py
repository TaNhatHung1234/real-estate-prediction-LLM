import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def train_and_save_model():
    print("1. Đang nạp và làm sạch dữ liệu...")
    df = pd.read_csv('vietnam_housing_dataset.csv')

    # Chuẩn hóa tên Tỉnh thành
    df['Tỉnh thành'] = df['Tỉnh thành'].astype(str).str.strip().str.rstrip('.')
    province_fix = {
        'TP. HCM': 'Hồ Chí Minh', 'TPHCM': 'Hồ Chí Minh', 'TpHCM': 'Hồ Chí Minh', 
        'TP Hồ Chí Minh': 'Hồ Chí Minh', 'Hồ Chí Mính': 'Hồ Chí Minh', 
        'HN': 'Hà Nội', 'Hà Nội': 'Hà Nội'
    }
    df['Tỉnh thành'] = df['Tỉnh thành'].replace(province_fix)
    
    valid_provinces = df['Tỉnh thành'].value_counts()[df['Tỉnh thành'].value_counts() > 10].index
    df_clean = df[df['Tỉnh thành'].isin(valid_provinces)].copy()

    # Features & Target
    X = df_clean[['Tỉnh thành', 'Quận huyện', 'Diện tích', 'Mặt tiền', 'Đường vào', 
                  'Số tầng', 'Số phòng ngủ', 'Số toilet', 'Pháp lý', 
                  'Nội thất', 'Hướng nhà']]
    y = df_clean['Giá bán']

    num_cols = ['Diện tích', 'Mặt tiền', 'Đường vào', 'Số tầng', 'Số phòng ngủ', 'Số toilet']
    cat_cols = ['Tỉnh thành', 'Quận huyện', 'Pháp lý', 'Nội thất', 'Hướng nhà']

    # Pipeline tiền xử lý
    num_transformer = SimpleImputer(strategy='median')
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Chưa rõ')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    # Mô hình ML
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42
        ))
    ])

    print("2. Đang huấn luyện mô hình Gradient Boosting...")
    model.fit(X, y)
    
    # Lưu mô hình vào file
    joblib.dump(model, 'dss_house_pricing_model.pkl')
    print("-> Đã huấn luyện thành công và lưu vào 'dss_house_pricing_model.pkl'!\n")

def predict_house_price(user_house_dict):
    """Hàm chạy dự đoán và sinh ra kết quả JSON cho DSS"""
    model = joblib.load('dss_house_pricing_model.pkl')
    input_df = pd.DataFrame([user_house_dict])
    predicted_price = model.predict(input_df)[0] # Tỷ VNĐ
    
    return {
        "thong_tin_nha_nguoi_ban": user_house_dict,
        "ket_qua_dinh_gia_ml": {
            "gia_du_doan_trung_vi_ty": round(predicted_price, 2),
            "kich_ban_gia_de_xuat": {
                "ban_nhanh_1_2_tuan": f"{round(predicted_price * 0.95, 2)} - {round(predicted_price * 0.98, 2)} tỷ VNĐ",
                "gia_thi_truong_1_thang": f"{round(predicted_price * 0.99, 2)} - {round(predicted_price * 1.02, 2)} tỷ VNĐ",
                "ky_vong_cao_2_3_thang": f"{round(predicted_price * 1.03, 2)} - {round(predicted_price * 1.06, 2)} tỷ VNĐ"
            }
        }
    }

if __name__ == "__main__":
    train_and_save_model()