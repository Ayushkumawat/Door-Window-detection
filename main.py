from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO
import uvicorn
from typing import List
from pydantic import BaseModel
import os
import sys
from pathlib import Path


app = FastAPI(
    title="Door and Window Detection API",
    description="API for detecting doors and windows in architectural blueprints using YOLOv8",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[float]

class DetectionResponse(BaseModel):
    detections: List[Detection]


try:
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    model = None

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    """
    Detect doors and windows in architectural blueprint images
    """
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        results = model(image)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = results.names[class_id]
            
            width = x2 - x1
            height = y2 - y1
            
            detections.append(Detection(
                label=label,
                confidence=round(confidence, 3),
                bbox=[float(x1), float(y1), float(width), float(height)]
            ))
        
        return DetectionResponse(detections=detections)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for all unhandled exceptions"""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )
