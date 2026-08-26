from sqlalchemy import Column, Integer, String, Float, DateTime
import datetime
from app.database import Base

class HazardReport(Base):
    __tablename__ = "hazard_reports"

    id = Column(Integer, primary_key=True, index=True)
    location_name = Column(String, index=True, nullable=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    corridor_name = Column(String, index=True, nullable=False)
    severity = Column(String, nullable=False)
    risk_score = Column(Integer, nullable=False)
    message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)