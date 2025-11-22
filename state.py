# app/state.py

from typing import Dict, List, Optional
from datetime import datetime
from .schemas import ZoneStatus

ZONE_STATE: Dict[str, ZoneStatus] = {}


def update_zone_status(
    zone_id: str,
    estimated_people: int,
    density_level: str,
    temperature: float | None = None,
    queue_length: int | None = None,
) -> None:
    density_factor = {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.8,
        "critical": 1.0,
    }.get(density_level, 0.5)

    temp_factor = 0.0
    if temperature is not None:
        if temperature > 34:
            temp_factor = 0.3
        elif temperature > 30:
            temp_factor = 0.15

    queue_factor = 0.0
    if queue_length is not None:
        if queue_length > 40:
            queue_factor = 0.3
        elif queue_length > 20:
            queue_factor = 0.15

    risk_score = min(1.0, density_factor + temp_factor + queue_factor)

    status = ZoneStatus(
        zone_id=zone_id,
        last_updated=datetime.utcnow(),
        estimated_people=estimated_people,
        density_level=density_level,
        temperature=temperature,
        queue_length=queue_length,
        risk_score=risk_score,
    )
    ZONE_STATE[zone_id] = status


def get_all_zones() -> List[ZoneStatus]:
    return list(ZONE_STATE.values())


def get_zone(zone_id: str) -> Optional[ZoneStatus]:
    return ZONE_STATE.get(zone_id)