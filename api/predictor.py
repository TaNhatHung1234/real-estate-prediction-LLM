import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT))

from ml_engine import predict_house_price


def predict(data):

    user_house = {
        "Tỉnh thành": data.province,
        "Quận huyện": data.district,
        "Diện tích": data.area,
        "Mặt tiền": data.frontage,
        "Đường vào": data.access_road,
        "Số tầng": data.floors,
        "Số phòng ngủ": data.bedrooms,
        "Số toilet": data.bathrooms,
        "Pháp lý": data.legal_status,
        "Nội thất": data.furniture,
        "Hướng nhà": data.direction
    }

    return predict_house_price(user_house)
