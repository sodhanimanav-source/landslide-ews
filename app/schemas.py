from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class HazardReportCreate(BaseModel):
    location_name: Optional[str] = None
    latitude: float
    longitude: float
    description: Optional[str] = None

class HazardReportResponse(BaseModel):
    id: int
    location_name: Optional[str] = None
    latitude: float
    longitude: float
    description: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True  # 'orm_mode = True' ko rename kiya gaya hai