from fastapi import FastAPI

from schemas import PredictionRequest
from predictor import predict_price

app = FastAPI(
    title="Real Estate DSS API"
)

@app.get("/")
def root():
    return {
        "message": "API Running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict(data: PredictionRequest):

    result = predict_price(data)

    return {
        "predicted_price": result
    }
