# 🏡 DSS Hỗ Trợ Định Giá & Tư Vấn Bất Động Sản

## Demo Online

Ứng dụng đã được triển khai trên Streamlit Cloud:

https://real-estate-prediction-llm-mkhcnb5vsplay23eicsbrv.streamlit.app

---

## Giao diện hệ thống

<img width="1777" height="858" alt="image" src="https://github.com/user-attachments/assets/d5e83695-eb7f-4e67-adc1-88083154a2c5" />

---

# Tổng quan dự án

Hệ thống Hỗ Trợ Quyết Định (Decision Support System - DSS) dành cho người bán bất động sản, kết hợp:

- Machine Learning Price Prediction
- Retrieval-Augmented Generation (RAG)
- Gemini AI Advisory Assistant
- FastAPI Prediction Service
- Streamlit Frontend

Mục tiêu của hệ thống là hỗ trợ người bán định giá tài sản, tham khảo dữ liệu thị trường tương đồng và nhận tư vấn từ AI dựa trên dữ liệu thực tế.

---

# Kiến trúc hệ thống

```text
User
│
▼
Streamlit Web Application
│
├── DSS Interface
├── RAG Retrieval Engine
└── Gemini AI Assistant
│
▼
FastAPI Prediction Service
│
▼
Gradient Boosting Regressor
│
▼
Housing Dataset
```

---

# Chức năng chính

✅ Dự đoán giá bất động sản bằng Machine Learning

✅ Sinh các kịch bản giá bán

- Bán nhanh
- Giá thị trường
- Giá kỳ vọng

✅ Truy xuất các bất động sản tương đồng từ dataset (RAG)

✅ Gemini AI Advisory Chatbot

✅ Guardrails hạn chế Hallucination

✅ REST API thông qua FastAPI

✅ Swagger/OpenAPI Documentation

---

# Công nghệ sử dụng

### Machine Learning

- Scikit-learn
- Gradient Boosting Regressor
- Pandas
- NumPy
- Joblib

### AI & LLM

- Google Gemini API
- Prompt Engineering
- Retrieval-Augmented Generation (RAG)

### Backend

- FastAPI
- Uvicorn

### Frontend

- Streamlit

---

# API Service

## Chạy FastAPI

```bash
uvicorn api.main:app --reload
```

Sau khi khởi động:

```text
http://127.0.0.1:8000/docs
```

Swagger UI sẽ tự động sinh tài liệu API.

---

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

---

## Predict House Price

```http
POST /predict
```

Request Example:

```json
{
  "province": "Hồ Chí Minh",
  "district": "Bình Chánh",
  "area": 85,
  "frontage": 5,
  "access_road": 6,
  "floors": 3,
  "bedrooms": 3,
  "bathrooms": 3,
  "legal_status": "Có sổ",
  "furniture": "Đầy đủ",
  "direction": "Nam"
}
```

---

# Cấu trúc thư mục

```text
real-estate-prediction-LLM
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
│
├── ai/
│
├── data/
│   ├── vietnam_housing_dataset.csv
│   └── chotot_test_dataset.csv│
│
├── docs/
│   ├── gemini-prompt-log.md
│   └── system-architecture.md
│
├── dss_house_pricing_model.pkl
├── app_dss.py
├── ml_engine.py
├── requirements.txt
└── README.md
```

---
---

# Đánh giá mô hình định giá

Dữ liệu được làm sạch trước khi chia: bỏ các dòng thiếu giá bán và các dòng trùng lặp, sau đó chia 80% train / 20% test với `random_state=42`. Mô hình chỉ được huấn luyện trên tập train.

## Kết quả trên tập test (5.460 căn)

| Mô hình | RMSE (tỷ) | MAE (tỷ) | R² | MAPE |
|---|---|---|---|---|
| Gradient Boosting (đang dùng) | 1,726 | 1,377 | 0,375 | 29,1% |
| Baseline: trung vị giá theo quận | 1,961 | 1,553 | 0,194 | 33,1% |

Mô hình cải thiện so với baseline trên cả bốn metric, nhưng mức cải thiện không lớn. R² = 0,375 nghĩa là mô hình giải thích khoảng 37% độ biến thiên giá. Sai số trung bình (MAE) khoảng 1,38 tỷ, tương đương khoảng 23% giá trung vị của dữ liệu. Dữ liệu được thu thập từ tin rao và thiếu nhiều thông tin quan trọng như vị trí chi tiết, tình trạng nhà và thời điểm giao dịch, nên giới hạn này chủ yếu đến từ dữ liệu.

## Tái lập

```bash
python -m evaluation.ml_eval
```

Kết quả được ghi vào `evaluation/results/ml_metrics.json`.

## Hạn chế

- Nhãn giá là giá rao bán, không phải giá giao dịch thực tế.
- Chưa đánh giá trên dữ liệu thu thập ở thời điểm khác (kiểm tra ngoài mẫu theo thời gian).
- Một số đặc trưng như Pháp lý và Nội thất có mức độ đóng góp rất thấp vào dự đoán của mô hình hiện tại.

---

# Hướng phát triển

### Sprint 1 ✅

- FastAPI Prediction Service
- Swagger Documentation

### Sprint 2 🚧

- Streamlit gọi FastAPI thay vì gọi model trực tiếp

### Sprint 3 🚧

- PostgreSQL Integration

### Sprint 4 🚧

- Text-to-SQL bằng Gemini

### Sprint 5 🚧

- Docker Deployment

### Sprint 6 🚧

- Hybrid Search (BM25 + BGE-M3)

---


# Tài liệu

- docs/gemini-prompt-log.md
- docs/system-architecture.md

---

# Phiên bản

Current Release: v0.2-fastapi

---

# Kỹ năng thể hiện trong dự án

- Machine Learning
- FastAPI
- REST API
- Streamlit
- Gemini API
- RAG
- Prompt Engineering
- API Design
- Model Serving
- AI Application Development
