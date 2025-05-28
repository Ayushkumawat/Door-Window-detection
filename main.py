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

# Get the port from environment variable for Render
PORT = int(os.getenv("PORT", 10000))

# Initialize FastAPI app
app = FastAPI(
    title="Door and Window Detection API",
    description="API for detecting doors and windows in architectural blueprints using YOLOv8",
    version="1.0.0",
)

# Configure CORS for all origins (you can restrict this in production)
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

# Model loading with error handling
try:
    # Get the absolute path to the model file
    model_path = Path(__file__).parent / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = YOLO(str(model_path))
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    model = None

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    """
    Detect doors and windows in architectural blueprint images
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Perform detection
        results = model(image)[0]
        
        # Process results
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = results.names[class_id]
            
            # Convert to [x, y, width, height] format
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

# For local development and Render deployment
if __name__ == "__main__":
    import uvicorn
    
    # Determine if running on Render
    is_render = os.getenv('RENDER') == 'true'
    
    # Configure host and port
    host = "0.0.0.0" if is_render else "127.0.0.1"
    port = PORT if is_render else 8000
    
    print(f"Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=not is_render  # Enable reload in development only
    )

swagger_file_path = os.path.join(os.path.dirname(__file__), "swagger.yaml")
try:
    with open(swagger_file_path, "r") as file:
        swagger_yaml = yaml.safe_load(file)
        
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        app.openapi_schema = swagger_yaml
        return app.openapi_schema

    app.openapi = custom_openapi
    print(f"Swagger documentation loaded from {swagger_file_path}")
except Exception as e:
    print(f"Error loading Swagger YAML: {e}")
