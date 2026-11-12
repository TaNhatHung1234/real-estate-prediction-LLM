from pydantic import BaseModel

class HouseInput(BaseModel):
    province: str
    district: str
    area: float
    frontage: float
    access_road: float
    floors: float
    bedrooms: float
    bathrooms: float
    legal_status: str
    furniture: str
    direction: str
