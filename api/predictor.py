import joblib
import numpy as np

model = joblib.load(
    "../dss_house_pricing_model.pkl"
)

def predict_price(data):

    X = np.array([[
        data.area,
        data.bedrooms,
        data.bathrooms
    ]])

    prediction = model.predict(X)

    return float(prediction[0])
