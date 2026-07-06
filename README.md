# Rash Driver Detection System

An end-to-end computer vision pipeline that detects, tracks, and classifies surrounding vehicles as rash or safe drivers using dashcam footage. Output is a video with color-coded bounding boxes based on real-time behavior analysis.

Motivated by ADAS work at Mercedes-Benz R&D India.

## Pipeline

```
Dashcam Video → YOLOv8s Detection → DeepOCSORT Tracking → ReID Matcher → Risk Scoring → Annotated Output
```

## Features

- **Vehicle Detection** — YOLOv8s detects cars, motorcycles, buses, trucks in each frame
- **Multi-Object Tracking** — DeepOCSORT with observation-centric re-matching for crowded scenes
- **Lighting-Robust ReID** — OSNet embeddings + Lab ab-channel histograms survive overpass shadow transitions
- **Road-Ahead ROI** — trapezoidal filter ignores crossing traffic, oncoming lanes, and ego-car hood
- **Risk Scoring** — classifies each vehicle as safe / moderate / rash *(in progress)*
- **Color-coded output** — green / orange / red bounding boxes *(in progress)*

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| YOLOv8s (Ultralytics) | Vehicle detection |
| DeepOCSORT (boxmot) | Multi-object tracking |
| OSNet (boxmot ReID) | Appearance-based re-identification |
| OpenCV | Video I/O, annotation, Lab colour space |
| NumPy | Feature computation |
| PyTorch | LSTM risk classifier *(planned)* |

## Project Structure

```
rash-driving-detection/
├── main.py                  ← entry point
├── modules/
│   ├── detector.py          ← YOLOv8 + ROI filter
│   ├── tracker.py           ← DeepOCSORT wrapper
│   ├── reid_matcher.py      ← embedding cache + Lab histogram + ID remap gate
│   └── visualizer.py        ← bounding box drawing
└── frames/                  ← (gitignored) extracted debug frames
```

## Setup

```bash
git clone https://github.com/sahishnusuresh/rash-driving-detection.git
cd rash-driving-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install ultralytics opencv-python boxmot numpy
```

## Adding Your Dashcam Video

Dashcam videos are not included in this repo due to file size. Place your video anywhere and pass the path via `--input`:

```bash
python main.py --input /path/to/your/dashcam.mp4
```

The output video is saved to `tracked.mp4` by default. Override with `--output`:

```bash
python main.py --input dashcam.mp4 --output result.mp4
```

**Video requirements:**
- Format: `.mp4` (H.264 recommended)
- Resolution: any — tested on 3840×2160 (4K) and 1920×1080
- Orientation: forward-facing dashcam, mounted behind windshield
- Content: road footage with other vehicles visible ahead

**Free dashcam footage sources if you don't have your own:**
- [Pexels](https://www.pexels.com/search/videos/dashcam/) — free, no license issues
- [BDD100K](https://bdd-data.berkeley.edu/) — research dataset, 100K dashcam clips

## Usage

```bash
# activate venv first
source venv/bin/activate

# run the tracker — output saved to tracked.mp4
python main.py --input dashcam.mp4

# save to a custom output path
python main.py --input dashcam.mp4 --output result.mp4
```

Output video `tracked.mp4` will appear in the project root with bounding boxes and tracker IDs.

**To extract frames from your original dashcam video (for inspection):**
```bash
mkdir frames

# 1 frame per second
ffmpeg -i dashcam1.mp4 -vf fps=1 frames/frame_%04d.jpg

# 2 frames per second (finer resolution — recommended for debugging)
ffmpeg -i dashcam1.mp4 -vf fps=2 frames/frame_%04d.jpg

# extract every frame (warning: large number of files on long videos)
ffmpeg -i dashcam1.mp4 frames/frame_%04d.jpg
```

**To extract frames from the tracked output video:**
```bash
ffmpeg -i tracked.mp4 -vf fps=2 frames/frame_%04d.jpg
```

## Known Limitations

- **Overpass / tunnel ID switches** — abrupt lighting changes cause appearance embeddings to drift. The ReID matcher catches most cases but the shadow-to-sunlight transition at overpass exit can still cause a 1–2 frame ID switch. Mitigation: position-based risk label persistence (planned for Week 5).
- **ID switches in crowded intersections** — vehicles with similar appearance (dark sedans) can swap IDs when boxes heavily overlap. DeepOCSORT's observation-centric re-matching reduces but does not eliminate this.
- **Far-away vehicle detection** — vehicles beyond ~40m are filtered out by the size threshold to reduce noise.

## Project Status

| Week | Feature | Status |
|------|---------|--------|
| 1 | YOLOv8 vehicle detection | Done |
| 2 | DeepOCSORT tracking + ReID matcher | Done |
| 3–4 | Optical flow speed estimation + VehicleTracker | In progress |
| 5–6 | LSTM risk classifier + color-coded visualization | Planned |
| 7–8 | Polish + demo video | Planned |

## Datasets

| Dataset | Use |
|---------|-----|
| BDD100K | Real dashcam videos with annotations |
| KITTI Tracking | Vehicle tracking benchmark |
| CARLA Simulator | Synthetic data generation |

## Future Scope

- Lane discipline detection using UFLD model
- Replace rule-based scorer with LSTM classifier trained on BDD100K
- License plate OCR for identity-stable tracking across lighting transitions
- Edge deployment on Jetson Nano
- Map visualization using Folium + GPS metadata
