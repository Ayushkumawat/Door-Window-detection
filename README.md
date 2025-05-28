# Door-Window-detection

This project implements an AI model to detect doors and windows in architectural blueprints using YOLOv8.

## Project Structure
```
├── images/           # Blueprint images for training
├── labels/          # YOLO format annotation files
├── classes.txt      # Class definitions
├── app.py          # FastAPI application
├── best.pt         # Trained model weights
└── README.md       # This file
```

## Setup Instructions

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install ultralytics fastapi python-multipart uvicorn opencv-python numpy
```

3. Run the API:
```bash
python app.py
```

The API will be available at `http://localhost:8000`

## API Usage

### POST /detect
Endpoint for detecting doors and windows in blueprint images.

#### Curl Example:
```bash
curl -X POST "http://localhost:8000/detect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/blueprint.jpg"
```

#### Response Format:
```json
{
  "detections": [
    {
      "label": "door",
      "confidence": 0.91,
      "bbox": [x, y, width, height]
    },
    {
      "label": "window",
      "confidence": 0.84,
      "bbox": [x, y, width, height]
    }
  ]
}
```

## Model Training

The model was trained on manually labeled architectural blueprints using YOLOv8. The training process included:
1. Manual labeling using LabelImg
2. Dataset split (80% train, 20% validation)
3. Training with YOLOv8 for optimal detection

## Training Screenshots

Example loss graph and console output from training:

![Screenshot 2025-05-28 204637](https://github.com/user-attachments/assets/33e25685-3403-4bcf-b737-446599922c5f)

![results](https://github.com/user-attachments/assets/a7a381ee-01de-475f-b1fd-5b6a0bacf469)

## Labeling Screenshots

Labeling with LabelImg and sample annotation files:

![Screenshot 2025-05-28 174326](https://github.com/user-attachments/assets/b1d25240-94ba-422e-a1dc-dff687f2f371)

### Sample Label Files

`7_png.rf.36887e50bc586c3daa12b06b13da870b.txt`:
```
1 0.153125 0.1078125 0.06640625 0.02109375
1 0.45078125 0.190625 0.02421875 0.1078125
1 0.3421875 0.47109375 0.13125 0.02578125
1 0.55546875 0.2921875 0.01875 0.0796875
1 0.9375 0.290625 0.015625 0.10859375
```

`6_png.rf.6968adaad271e77eb99fdf5710f5e46d.txt`:
```
0 0.31875 0.36953125 0.06640625 0.1484375
0 0.25625 0.4875 0.09140625 0.1109375
0 0.44921875 0.4671875 0.065625 0.109375
0 0.53828125 0.50390625 0.08046875 0.12265625
0 0.89296875 0.434375 0.08515625 0.11328125
```

Each line: `<class_id> <x_center> <y_center> <width> <height>` (normalized, YOLO format)

## Classes Used

The model detects the following classes:

- `door`: Door symbols in blueprints
- `window`: Window symbols in blueprints 
