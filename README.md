# Rash Driver Detection System

An end-to-end computer vision pipeline that detects, tracks, and classifies surrounding vehicles as rash or safe drivers using dashcam footage. Output is a video with color-coded bounding boxes (green = safe, yellow = moderate, red = rash) based on real-time behavior analysis.

Motivated by ADAS work at Mercedes-Benz R&D India.

## Pipeline

```
Dashcam Video → YOLOv8 Detection → ByteTrack Tracking → Feature Extraction → Risk Scoring → Annotated Output
```

## Features

- **Vehicle Detection** — YOLOv8 detects cars, motorcycles, buses, trucks in each frame
- **Multi-Object Tracking** — ByteTrack assigns persistent IDs across frames
- **6 Rash Driving Behaviors Detected:**
  - Sudden braking
  - Tailgating
  - Lane weaving
  - Sudden cut-ins
  - Overspeeding (relative)
  - Aggressive acceleration
- **Risk Scoring** — rule-based scorer classifies each vehicle as safe / moderate / rash
- **Color-coded output** — green / orange / red bounding boxes with live HUD overlay
- **LLM Alerts** — spoken natural language alerts via Ollama + pyttsx3 when a vehicle turns rash
- **Post-trip Reports** — LLM-generated safety report saved to a route risk database after each run
- **Route Queries** — ask "How risky is MG Road?" and get a summary across past trips

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| YOLOv8 (Ultralytics) | Vehicle detection |
| ByteTrack (Supervision) | Multi-object tracking |
| OpenCV | Video I/O and annotation |
| NumPy / Pandas | Feature computation |
| Ollama (LLaMA 3.2) | Alert generation and trip reports |
| pyttsx3 | Text-to-speech alerts |

## Setup

```bash
git clone https://github.com/sahishnusuresh/rash-driving-detection.git
cd rash-driving-detection
pip install ultralytics opencv-python supervision numpy pandas ollama pyttsx3
```

## Usage

```bash
# Week 1 — basic detection
python detect.py

# Full pipeline (Week 5+ onwards)
python main.py --input dashcam.mp4 --output output.mp4 --route "MG Road Bangalore"

# Query route risk
python query_route.py --route "MG Road"
```

## Project Status

| Week | Feature | Status |
|------|---------|--------|
| 1 | YOLOv8 vehicle detection | Done |
| 2 | ByteTrack multi-object tracking | In progress |
| 3–4 | Feature extraction (6 behaviors) | Planned |
| 5–6 | Risk scoring + color-coded visualization | Planned |
| 5–6 | LLM alerts via Ollama | Planned |
| 7 | Post-trip reports + route risk database | Planned |
| 8 | Polish + demo video | Planned |

## Datasets

| Dataset | Use |
|---------|-----|
| BDD100K | Real dashcam videos with annotations |
| KITTI Tracking | Vehicle tracking benchmark |
| CARLA Simulator | Synthetic data generation |

## Future Scope

- Lane discipline detection using UFLD model
- Map visualization using Folium + GPS metadata
- Replace rule-based scorer with LSTM classifier trained on BDD100K
- Edge deployment on Jetson Nano
