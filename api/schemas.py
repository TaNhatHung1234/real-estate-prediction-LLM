from pydantic import BaseModel

class PredictionRequest(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
