from fastapi import FastAPI

from api.schemas import HouseInput
from api.predictor import predict


app = FastAPI(
    title="Real Estate DSS API",
    version="1.0"
)
#Health check:
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

#AI dự đoán:
@app.post("/predict")
def predict_house(data: HouseInput):

    result = predict(data)

    return result
