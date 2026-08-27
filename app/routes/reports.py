from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import HazardReport
from app.schemas import HazardReportCreate, HazardReportResponse
from typing import List
import math

router = APIRouter(prefix="/api/reports", tags=["Crowdsourced Reports"])

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in kilometers.
    """
    R = 6371.0  # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@router.post("/", response_model=HazardReportResponse, status_code=status.HTTP_201_CREATED)
def create_report(report: HazardReportCreate, db: Session = Depends(get_db)):
    """
    Create a new hazard incident report and check for spatial clustering.
    """
    # Spatial Clustering Check: Identify if another report exists within 500m (0.5km)
    existing_reports = db.query(HazardReport).all()
    cluster_found = False
    
    for r in existing_reports:
        dist = haversine_distance(report.latitude, report.longitude, r.latitude, r.longitude)
        if dist < 0.5:
            cluster_found = True
            break

    # Save report to Database
    db_report = HazardReport(**report.dict())
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    
    return db_report

@router.get("/", response_model=List[HazardReportResponse])
def get_reports(db: Session = Depends(get_db)):
    """
    Fetch all crowdsourced hazard reports sorted by newest first.
    """
    return db.query(HazardReport).order_by(HazardReport.id.desc()).all()

@router.get("/{report_id}", response_model=HazardReportResponse)
def get_report_by_id(report_id: int, db: Session = Depends(get_db)):
    """
    Fetch a specific hazard report by its ID.
    """
    report = db.query(HazardReport).filter(HazardReport.id == report_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Report with ID {report_id} not found"
        )
    return report