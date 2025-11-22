# app/schemas.py

from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class IoTIngestRequest(BaseModel):
    zone_id: str = Field(..., example="zone_a")
    people_count: Optional[int] = Field(None, example=42)
    temperature: Optional[float] = Field(None, example=28.5)
    queue_length: Optional[int] = Field(None, example=15)
    extra: Optional[dict] = None


class ZoneStatus(BaseModel):
    zone_id: str
    last_updated: datetime
    estimated_people: int
    density_level: str  # "low", "medium", "high", "critical"
    temperature: Optional[float] = None
    queue_length: Optional[int] = None
    risk_score: float


class Recommendation(BaseModel):
    zone_id: str
    message: str
    priority: str  # "low", "medium", "high", "critical"


class DashboardResponse(BaseModel):
    zones: List[ZoneStatus]
    recommendations: List[Recommendation]