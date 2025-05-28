from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO
import uvicorn
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[float]

class DetectionResponse(BaseModel):
    detections: List[Detection]

app = FastAPI(title="Door and Window Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model globally - update path to where your best.pt is located
model = YOLO("runs/detect/train/weights/best.pt")

def test_on_image(image_path):
    """
    Test the model on a single image and display results
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Run inference
    results = model(image)[0]
    
    # Draw results on image
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = results.names[class_id]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(image, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the result
    output_path = f"results_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, image)
    print(f"Results saved to {output_path}")
    
    # Print detections
    print("\nDetections:")
    for box in results.boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = results.names[class_id]
        print(f"{label}: {confidence:.2f}")

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    # Read and process the image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
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

if __name__ == "__main__":
    # Test mode: If you want to test on specific images
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test on images in the test directory
        test_dir = "testing"
        if os.path.exists(test_dir):
            print(f"Testing on images in {test_dir}")
            for image_file in os.listdir(test_dir):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(test_dir, image_file)
                    print(f"\nProcessing {image_file}...")
                    test_on_image(image_path)
        else:
            print("Please provide test images in the test directory")
    else:
        # Run as API server
        uvicorn.run(app, host="0.0.0.0", port=8000)

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