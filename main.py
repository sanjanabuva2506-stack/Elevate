# app/main.py

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

import numpy as np
import cv2

from .schemas import IoTIngestRequest, DashboardResponse
from .state import update_zone_status, get_all_zones, get_zone
from .processing import load_model, estimate_crowd_from_frame
from .strategy import generate_recommendations
from . import kafka_client


app = FastAPI(title="AI Strategy Planner", version="1.0.0")

templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    load_model()
    print("AI Strategy Planner API started")


@app.get("/")
def root():
    return {"message": "AI Strategy Planner backend running", "ui": "/ui"}


@app.get("/ui", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.post("/ingest/frame", summary="Ingest a frame from camera")
async def ingest_frame(
    zone_id: str,
    file: UploadFile = File(...),
):
    """
    Accepts an image (JPEG/PNG), estimates crowd, updates zone state.
    """
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    estimated_people, density_level = estimate_crowd_from_frame(frame)

    existing = get_zone(zone_id)
    temperature = existing.temperature if existing else None
    queue_length = existing.queue_length if existing else None

    update_zone_status(
        zone_id=zone_id,
        estimated_people=estimated_people,
        density_level=density_level,
        temperature=temperature,
        queue_length=queue_length,
    )

    kafka_client.publish("camera_frames", {
        "zone_id": zone_id,
        "estimated_people": estimated_people,
        "density_level": density_level,
    })

    return {
        "zone_id": zone_id,
        "estimated_people": estimated_people,
        "density_level": density_level,
    }


@app.post("/ingest/iot", summary="Ingest IoT/sensor data")
def ingest_iot(payload: IoTIngestRequest):
    existing = get_zone(payload.zone_id)

    if existing:
        estimated_people = existing.estimated_people
        density_level = existing.density_level
    else:
        estimated_people = payload.people_count or 0
        density_level = "low"

    # override crowd if sensor gives direct people count
    if payload.people_count is not None:
        estimated_people = payload.people_count
        if estimated_people < 20:
            density_level = "low"
        elif estimated_people < 50:
            density_level = "medium"
        elif estimated_people < 100:
            density_level = "high"
        else:
            density_level = "critical"

    update_zone_status(
        zone_id=payload.zone_id,
        estimated_people=estimated_people,
        density_level=density_level,
        temperature=payload.temperature,
        queue_length=payload.queue_length,
    )

    kafka_client.publish("iot_events", {
        "zone_id": payload.zone_id,
        "people_count": payload.people_count,
        "temperature": payload.temperature,
        "queue_length": payload.queue_length,
    })

    return {"status": "ok", "zone_id": payload.zone_id}


@app.get("/dashboard", response_model=DashboardResponse)
def get_dashboard():
    zones = get_all_zones()
    recs = generate_recommendations(zones)

    kafka_client.publish("zone_status", {
        "zones": [z.dict() for z in zones],
        "recommendations": [r.dict() for r in recs],
    })

    return DashboardResponse(zones=zones, recommendations=recs)