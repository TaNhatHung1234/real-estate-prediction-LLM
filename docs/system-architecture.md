# System Architecture

## Overview

Hệ thống DSS hỗ trợ định giá và tư vấn bán bất động sản được xây dựng dựa trên kiến trúc gồm giao diện người dùng, mô hình Machine Learning và trợ lý AI Gemini.

## Components

### 1. Streamlit Frontend

Chức năng:

- Nhập thông tin bất động sản
- Hiển thị kết quả định giá
- Hiển thị tư vấn AI
- Nhập Gemini API Key

### 2. Machine Learning Engine

Tệp chính:

- ml_engine.py
- dss_house_pricing_model.pkl

Chức năng:

- Tiền xử lý dữ liệu
- Dự đoán giá bất động sản
- Sinh các kịch bản giá bán

Đầu ra:

- Giá trung vị
- Giá bán nhanh
- Giá thị trường
- Giá kỳ vọng

### 3. Gemini Advisor

Chức năng:

- Trả lời câu hỏi của người dùng
- Phân tích kết quả định giá
- Tư vấn bán bất động sản

Cơ chế bảo vệ:

- Guardrails
- Giới hạn theo dữ liệu định giá
- Hạn chế hallucination

### 4. Deployment

Source Code:
- GitHub

Hosting:
- Streamlit Cloud

## Data Flow

User
↓
Input Form
↓
Machine Learning Engine
↓
Price Prediction
↓
Pricing Scenarios
↓
Gemini Advisor
↓
Recommendation
↓
User

## Technologies

- Python
- Streamlit
- Scikit-learn
- Pandas
- Joblib
- Google Gemini API
- GitHub
- Streamlit Cloud
