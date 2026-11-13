"""Đánh giá mô hình giá trên tập test 20% và so sánh với baseline.
Chạy từ thư mục gốc repo: python -m evaluation.ml_eval
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "vietnam_housing_dataset.csv"
OUT = ROOT / "evaluation" / "results" / "ml_metrics.json"

TARGET = "Giá bán"
NUM = ["Diện tích", "Mặt tiền", "Đường vào", "Số tầng", "Số phòng ngủ", "Số toilet"]
CAT = ["Tỉnh thành", "Quận huyện", "Pháp lý", "Nội thất", "Hướng nhà"]
FEATURES = CAT + NUM
PROVINCE_FIX = {
    "TP. HCM": "Hồ Chí Minh", "TPHCM": "Hồ Chí Minh", "TpHCM": "Hồ Chí Minh",
    "TP Hồ Chí Minh": "Hồ Chí Minh", "Hồ Chí Mính": "Hồ Chí Minh", "HN": "Hà Nội",
}


def load():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df["Tỉnh thành"] = (df["Tỉnh thành"].astype(str).str.strip()
                        .str.rstrip(".").replace(PROVINCE_FIX))
    df = df.dropna(subset=[TARGET])        # bắt buộc: metric không chấp nhận NaN
    df = df.drop_duplicates()              # tránh cùng một căn nằm cả ở train và test
    counts = df["Tỉnh thành"].value_counts()
    return df[df["Tỉnh thành"].isin(counts[counts > 10].index)].copy()


def build_pipeline():
    # Giống hệt cấu hình trong ml_engine.py để đánh giá đúng model đang deploy
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Chưa rõ")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CAT),
    ])
    return Pipeline([
        ("preprocessor", pre),
        ("regressor", GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)),
    ])


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "n_test": int(len(y_true)),
        "RMSE_ty": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "MAE_ty": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MAPE_pct": round(float(np.mean(np.abs(y_true - y_pred) / y_true) * 100), 2),
    }


def baseline_predict(train, test):
    """Trung vị giá theo quận; thiếu quận thì lấy trung vị tỉnh; cuối cùng là trung vị toàn cục."""
    by_dist = train.groupby(["Tỉnh thành", "Quận huyện"])[TARGET].median()
    by_prov = train.groupby("Tỉnh thành")[TARGET].median()
    glob = train[TARGET].median()
    out = []
    for p, d in zip(test["Tỉnh thành"], test["Quận huyện"]):
        if (p, d) in by_dist.index:
            out.append(by_dist[(p, d)])
        elif p in by_prov.index:
            out.append(by_prov[p])
        else:
            out.append(glob)
    return np.array(out)


def main():
    df = load()
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    model = build_pipeline().fit(train[FEATURES], train[TARGET])
    y = test[TARGET].to_numpy()
    results = {
        "n_total_after_clean": int(len(df)),
        "gradient_boosting": metrics(y, model.predict(test[FEATURES])),
        "baseline_district_median": metrics(y, baseline_predict(train, test)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
