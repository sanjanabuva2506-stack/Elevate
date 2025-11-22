# app/strategy.py

from typing import List
from .schemas import ZoneStatus, Recommendation


def max_priority(p1: str, p2: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    return p1 if order[p1] >= order[p2] else p2


def generate_recommendations(zones: List[ZoneStatus]) -> List[Recommendation]:
    recommendations: List[Recommendation] = []

    for z in zones:
        msg_parts: list[str] = []
        priority = "low"

        # Base on density level
        if z.density_level == "critical":
            msg_parts.append(
                "Critical crowding detected. Deploy staff, redirect crowd, and temporarily restrict entry."
            )
            priority = "critical"
        elif z.density_level == "high":
            msg_parts.append(
                "High density. Increase monitoring and open alternate paths."
            )
            priority = "high"
        elif z.density_level == "medium":
            msg_parts.append(
                "Medium crowd. Keep staff on standby and watch queues."
            )
            priority = "medium"

        # Temperature awareness
        if z.temperature is not None and z.temperature > 34:
            msg_parts.append(
                "High temperature. Arrange hydration points and shade."
            )
            priority = max_priority(priority, "high")

        # Queue awareness
        if z.queue_length is not None and z.queue_length > 30:
            msg_parts.append(
                "Long queues. Add more counters or staff to speed up flow."
            )
            priority = max_priority(priority, "high")

        if not msg_parts:
            msg_parts.append("Situation normal. Continue monitoring.")

        message = f"Zone {z.zone_id}: " + " ".join(msg_parts)

        recommendations.append(
            Recommendation(
                zone_id=z.zone_id,
                message=message,
                priority=priority,
            )
        )

    return recommendations