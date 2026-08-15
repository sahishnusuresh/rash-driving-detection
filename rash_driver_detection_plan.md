# Rash Driver Detection System — Project Plan
**Author:** Suresah (Mercedes-Benz R&D India)
**Start Date:** June 20, 2026
**Target Completion:** August 15, 2026
**Purpose:** Graduate school applications — Spring 2027 (Texas A&M, San Jose State)

---

## Project Summary

Build an end-to-end pipeline that detects, tracks, and classifies surrounding vehicles
as rash or safe drivers using dashcam footage. Output is a video with color-coded
bounding boxes (green = safe, yellow = moderate, red = rash) based on speed and
braking behavior.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| YOLOv8 (Ultralytics) | Vehicle detection |
| ByteTrack (Supervision) | Multi-object tracking |
| OpenCV | Video processing |
| NumPy / Pandas | Feature computation |

---

## Week-by-Week Plan

---

### Week 1 — Jun 20–27 | Environment + Detection

**Goal:** YOLOv8 detecting vehicles in a dashcam video

**Tasks:**
- [ ] Install dependencies
  ```bash
  pip install ultralytics opencv-python supervision yt-dlp numpy pandas
  ```
- [ ] Download a dashcam video (use any of these free sources)
  ```bash
  # Option 1 — YouTube (search "India dashcam rash driving")
  yt-dlp "https://www.youtube.com/watch?v=<video_id>" -o dashcam.mp4

  # Option 2 — Pexels / Pixabay / Videvo (free, no license issues)
  # Download manually from pexels.com/videos or pixabay.com/videos
  ```
  **Avoid Getty Images** — videos are licensed/copyrighted, preview only
- [ ] Run pretrained YOLOv8 on video
  ```python
  from ultralytics import YOLO
  model = YOLO('yolov8n.pt')
  model('dashcam.mp4', show=True, classes=[2, 3, 5, 7])  # cars, motorcycles, buses, trucks
  ```
- [ ] Understand: bounding boxes, confidence scores, class IDs
- [ ] Watch: CS231n Lectures 1–3 (youtube, free)
- [ ] Create GitHub repo, push first script

**Deliverable:** Video with bounding boxes around all vehicles

---

### Week 2 — Jun 28–Jul 4 | Object Tracking

**Goal:** Each vehicle gets a persistent ID across frames

**Tasks:**
- [ ] Integrate ByteTrack via Supervision library
  ```python
  import supervision as sv
  from ultralytics import YOLO

  model = YOLO('yolov8n.pt')
  tracker = sv.ByteTrack()
  annotator = sv.BoxAnnotator()

  cap = cv2.VideoCapture('dashcam.mp4')
  while cap.isOpened():
      ret, frame = cap.read()
      results = model(frame, classes=[2, 3, 5, 7])[0]
      detections = sv.Detections.from_ultralytics(results)
      detections = tracker.update_with_detections(detections)
      # detections.tracker_id now has persistent IDs
  ```
- [ ] Display tracker ID on each vehicle bounding box
- [ ] Read: SORT paper (8 pages, arxiv 1602.00763)
- [ ] Push to GitHub

**Deliverable:** Video with vehicle IDs that persist across frames

---

### Week 3–4 — Jul 5–18 | Feature Extraction

**Goal:** Compute all rash driving parameters per tracked vehicle from bounding boxes alone

**Per-vehicle state to track:**
```python
from collections import defaultdict, deque

vehicle_state = defaultdict(lambda: {
    'bbox_history':  [],        # list of bboxes per frame
    'cx_history':    [],        # centroid X history (for weaving)
    'speed_history': [],        # relative speed per frame
    'close_history': deque(maxlen=30),  # tailgating window
    'braking_count': 0,
    'weave_count':   0,
    'tailgating':    False,
    'cut_in':        False,
    'aggressive_acc':False,
    'frames_tracked': 0,
    'age':           0,         # frames since first detection
})
```

**Tasks:**

- [ ] **Relative Speed** — centroid displacement normalized by bbox height
  ```python
  def relative_speed(bbox_prev, bbox_curr):
      cx_p = (bbox_prev[0] + bbox_prev[2]) / 2
      cy_p = (bbox_prev[1] + bbox_prev[3]) / 2
      cx_c = (bbox_curr[0] + bbox_curr[2]) / 2
      cy_c = (bbox_curr[1] + bbox_curr[3]) / 2
      displacement = ((cx_c - cx_p)**2 + (cy_c - cy_p)**2) ** 0.5
      bbox_height = bbox_curr[3] - bbox_curr[1]
      return displacement / (bbox_height + 1e-5)  # scale-invariant units
  ```
  *Normalizing by bbox height corrects for perspective — far vehicles have small bboxes*

- [ ] **Sudden Braking** — speed drops sharply over 3 consecutive frames
  ```python
  def detect_braking(speed_history, threshold=0.08):
      if len(speed_history) < 3:
          return False
      return (speed_history[-3] - speed_history[-1]) > threshold
  ```
  *Speed was high, now suddenly low = braking event*

- [ ] **Tailgating** — bbox height occupies large fraction of frame for 30+ frames
  ```python
  def detect_tailgating(bbox, frame_height, close_history, threshold=0.4, min_frames=30):
      bbox_height = bbox[3] - bbox[1]
      close_history.append((bbox_height / frame_height) > threshold)
      return sum(close_history) >= min_frames
  ```
  *Large bbox sustained = physically close to camera for extended time*

- [ ] **Lane Weaving** — centroid X changes direction 3+ times in 15-frame window
  ```python
  def detect_weaving(cx_history, window=15, min_changes=3):
      if len(cx_history) < window:
          return False
      recent = cx_history[-window:]
      changes = sum(
          1 for i in range(1, len(recent)-1)
          if (recent[i] - recent[i-1]) * (recent[i+1] - recent[i]) < 0
      )
      return changes >= min_changes
  ```
  *Centroid X zigzags = vehicle weaving between lanes*

- [ ] **Sudden Cut-in** — new vehicle appears with already-large bbox near center
  ```python
  def detect_cut_in(bbox, frame_width, frame_height, age):
      if age > 5:
          return False
      bbox_height = bbox[3] - bbox[1]
      cx = (bbox[0] + bbox[2]) / 2
      is_large  = (bbox_height / frame_height) > 0.25
      is_center = (frame_width * 0.25) < cx < (frame_width * 0.75)
      return is_large and is_center
  ```
  *New tracker ID + already big bbox near center = cut in front of camera vehicle*

- [ ] **Aggressive Acceleration** — bbox shrinks rapidly over 3 frames
  ```python
  def detect_aggressive_acceleration(speed_history, threshold=-0.08):
      if len(speed_history) < 3:
          return False
      return (speed_history[-1] - speed_history[-3]) < threshold
  ```
  *Negative relative speed = vehicle pulling away fast*

- [ ] Update all 6 parameters per vehicle every frame in main loop
- [ ] Print feature summary per vehicle every 30 frames
- [ ] Push to GitHub

**Deliverable:** Console output showing all 6 parameters per vehicle ID per frame

---

### Week 5–6 — Jul 19–Aug 1 | Risk Scoring + Visualization

**Goal:** Color-code vehicles based on risk score in video output

**Tasks:**
- [ ] Build rule-based risk scorer
  ```python
  def compute_risk_score(avg_speed, braking_count, frames_tracked):
      if frames_tracked < 15:
          return 'unknown'  # not enough data

      score = 0
      if avg_speed > 50:
          score += 40
      elif avg_speed > 30:
          score += 20

      if braking_count >= 3:
          score += 40
      elif braking_count >= 1:
          score += 20

      if score >= 60:
          return 'rash'
      elif score >= 30:
          return 'moderate'
      return 'safe'
  ```
- [ ] Map risk to color
  ```python
  RISK_COLORS = {
      'safe':     (0, 255, 0),    # green
      'moderate': (0, 165, 255),  # orange
      'rash':     (0, 0, 255),    # red
      'unknown':  (200, 200, 200) # grey
  }
  ```
- [ ] Draw colored bounding boxes + risk label on each vehicle
- [ ] Draw HUD overlay showing live risk counts per frame
  ```python
  def draw_hud(frame, risk_counts):
      rash     = risk_counts.get('rash', 0)
      moderate = risk_counts.get('moderate', 0)
      safe     = risk_counts.get('safe', 0)

      cv2.rectangle(frame, (10, 10), (280, 70), (0, 0, 0), -1)  # black bg
      cv2.putText(frame, f"Rash: {rash}  Moderate: {moderate}  Safe: {safe}",
                  (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
  ```
  Build risk_counts by iterating vehicle_history and calling compute_risk_score each frame
- [ ] Write output video to file using cv2.VideoWriter
- [ ] Test on 3 different dashcam clips
- [ ] Push to GitHub

**Deliverable:** Full output video with green/orange/red labeled vehicles + live HUD counter

---

### Week 5–6 (Extension) | LLM Narration Alerts

**Goal:** Natural language alerts describing rash vehicle position and behavior

**Tasks:**
- [ ] Install dependencies
  ```bash
  # Install Ollama: https://ollama.com/download
  ollama pull llama3.2   # or mistral, phi3
  pip install ollama pyttsx3
  ```
- [ ] Detect spatial zone from bounding box position
  ```python
  def get_zone(bbox, frame_width):
      cx = (bbox[0] + bbox[2]) / 2
      if cx < frame_width * 0.33:
          return "left lane"
      elif cx < frame_width * 0.66:
          return "front"
      return "right lane"
  ```
- [ ] Build event log — trigger when a vehicle changes risk state to 'rash'
  ```python
  # e.g., {"tracker_id": 7, "zone": "front", "avg_speed": 62, "braking_count": 3}
  ```
- [ ] Generate alert via Ollama and speak via pyttsx3 — both in background threads
  ```python
  import threading
  import pyttsx3
  import ollama

  tts_engine = pyttsx3.init()

  def _speak(text):
      tts_engine.say(text)
      tts_engine.runAndWait()

  def speak_alert(text):
      threading.Thread(target=_speak, args=(text,), daemon=True).start()

  def generate_alert(event):
      prompt = (
          f"Vehicle #{event['tracker_id']} has been classified as a rash driver. "
          f"Zone: {event['zone']}. Relative speed: {event['avg_speed']:.1f}. "
          f"Braking events: {event['braking_count']}. "
          f"Generate a short, clear driver alert in 1 sentence only."
      )
      response = ollama.chat(
          model="llama3.2",
          messages=[{"role": "user", "content": prompt}]
      )
      return response['message']['content'].strip()

  def trigger_alert(event):
      def _run():
          alert = generate_alert(event)  # LLM call in background
          speak_alert(alert)             # audio plays when ready
          print(f"[ALERT] {alert}")
      threading.Thread(target=_run, daemon=True).start()
  ```
- [ ] In main loop, call `trigger_alert(event)` when a vehicle transitions to 'rash'
- [ ] Overlay alert text on `cv2.imshow` window for 3 seconds
- [ ] Rate-limit alerts: max 1 alert per vehicle per 5 seconds to avoid spam
- [ ] Make sure Ollama server is running before pipeline starts (`ollama serve`)
- [ ] Push to GitHub

**Note:** Video runs via `cv2.imshow` (real-time), audio plays through speakers. Output video is NOT saved with audio — for demo recording, use OBS or any screen recorder to capture both screen and speakers.

**Deliverable:** Live playback window with color-coded vehicles + spoken alerts like "Vehicle #7 approaching rapidly from the right lane"

---

### Week 7 — Aug 2–8 | Post-Trip Report + Route Risk Database

**Goal:** After each video, generate an LLM report and store it for route-level risk insights

**How it works:**
- Each trip report is saved as a JSON entry with a user-provided route/location tag
- Over multiple trips, the database builds a picture of which routes/areas are risky
- User can query: *"How risky is the MG Road route?"*

**Tasks:**

- [ ] Collect all detection events into a trip summary at end of video
  ```python
  def build_trip_summary(vehicle_state, video_path, route_tag, duration_secs):
      total    = len(vehicle_state)
      rash     = sum(1 for v in vehicle_state.values() if v['risk'] == 'rash')
      moderate = sum(1 for v in vehicle_state.values() if v['risk'] == 'moderate')
      events   = [
          {"id": tid, "risk": v['risk'], "braking": v['braking_count'],
           "tailgating": v['tailgating'], "weaving": v['weave_count'],
           "cut_in": v['cut_in']}
          for tid, v in vehicle_state.items() if v['risk'] != 'unknown'
      ]
      return {
          "video": video_path,
          "route": route_tag,
          "duration_secs": duration_secs,
          "total_vehicles": total,
          "rash": rash,
          "moderate": moderate,
          "safe": total - rash - moderate,
          "events": events
      }
  ```

- [ ] Send trip summary to Ollama to generate natural language report
  ```python
  def generate_trip_report(summary):
      prompt = f"""
  You are a road safety analyst. Given this trip summary, write a concise safety report (3-5 sentences).
  Mention total vehicles, rash driver count, most common violations, and overall risk level.

  Trip data: {summary}
  """
      response = ollama.chat(
          model="llama3.2",
          messages=[{"role": "user", "content": prompt}]
      )
      return response['message']['content'].strip()
  ```

- [ ] Save report + raw summary to route risk database (JSON file)
  ```python
  import json, os
  from datetime import datetime

  DB_PATH = "route_risk_db.json"

  def save_to_database(summary, report_text):
      db = []
      if os.path.exists(DB_PATH):
          with open(DB_PATH) as f:
              db = json.load(f)
      db.append({
          "timestamp": datetime.now().isoformat(),
          "route": summary['route'],
          "risk_level": "HIGH" if summary['rash'] >= 3 else
                        "MEDIUM" if summary['rash'] >= 1 else "LOW",
          "summary": summary,
          "report": report_text
      })
      with open(DB_PATH, 'w') as f:
          json.dump(db, f, indent=2)
  ```

- [ ] Add `--route` argument to main.py so user can tag each trip
  ```bash
  python main.py --input dashcam.mp4 --route "MG Road Bangalore"
  ```

- [ ] Add route query command — user asks about a route, LLM summarizes past trips
  ```bash
  python query_route.py --route "MG Road"
  # Output: "Based on 3 trips on MG Road, this route has HIGH risk.
  #          Common violations: tailgating (8 events), cut-ins (5 events)."
  ```
  ```python
  def query_route(route_name):
      with open(DB_PATH) as f:
          db = json.load(f)
      relevant = [e for e in db if route_name.lower() in e['route'].lower()]
      if not relevant:
          print(f"No data for route: {route_name}")
          return
      prompt = f"""
  Summarize road safety for the route "{route_name}" based on these past trip records.
  Mention number of trips, average risk level, most frequent violations, and a recommendation.

  Records: {relevant}
  """
      response = ollama.chat(model="llama3.2",
                             messages=[{"role": "user", "content": prompt}])
      print(response['message']['content'].strip())
  ```

- [ ] Print report to console after video finishes
- [ ] Save report as `reports/report_<timestamp>.txt` alongside database
- [ ] Push to GitHub

**Deliverable:** `route_risk_db.json` growing with each trip + queryable route risk summaries

---

### Week 8 — Aug 9–15 | Polish + Demo

**Goal:** Resume-ready GitHub project with demo video

**Tasks:**
- [ ] Clean up code — single main.py entry point
  ```
  python main.py --input dashcam.mp4 --output output.mp4 --route "Route Name"
  ```
- [ ] Write README.md with:
  - Project description
  - Demo GIF or video link
  - Installation instructions
  - How it works (pipeline diagram)
  - Sample results + sample report output
- [ ] Record 2-minute demo video showing:
  - Raw dashcam input
  - Detection + tracking with HUD
  - Risk-colored output + spoken alerts
  - Post-trip report printed to console
  - Route query example
- [ ] Upload demo to YouTube (unlisted), link in README
- [ ] Final GitHub push

**Deliverable:** Clean public GitHub repo ready to link on resume

---

## Resume Bullet Points (fill in after completion)

```
Rash Driver Detection System | Python, YOLOv8, ByteTrack, OpenCV, Ollama
- Built end-to-end pipeline detecting 6 rash driving behaviors (tailgating, cut-ins,
  weaving, sudden braking, overspeeding, aggressive acceleration) from dashcam footage
- Real-time risk scoring with color-coded bounding boxes, HUD overlay, and spoken
  LLM-generated alerts via Ollama + pyttsx3
- Post-trip safety reports generated by local LLM; stored in route risk database
  enabling historical route-level risk queries ("How risky is MG Road?")
- Motivated by ADAS work at Mercedes-Benz R&D India
- GitHub: github.com/<your_username>/rash-driver-detection
```

---

## SOP One-Liner

> "While building Android framework services for ADAS teams at Mercedes-Benz R&D India,
> I identified a gap in surrounding driver risk awareness and independently developed a
> real-time rash driver detection system using computer vision — which motivated my
> interest in pursuing graduate research in intelligent transportation systems."

---

## Datasets (for testing + future training)

| Dataset | Use | Link |
|---|---|---|
| BDD100K | Real dashcam videos with annotations | bdd-data.berkeley.edu |
| KITTI Tracking | Vehicle tracking benchmark | cvlibs.net/datasets/kitti |
| CARLA Simulator | Synthetic data generation | carla.org |

---

## Future Scope (post-August, mention in SOP)

- Lane discipline detection using UFLD model
- Map visualization using Folium + GPS
- Replace rule-based scorer with LSTM classifier trained on BDD100K
- Edge deployment on embedded hardware (Jetson Nano)

---

## Daily Habit

- Minimum 2 hrs/day on weekdays
- 3–4 hrs on weekends
- Every Sunday: push code to GitHub + note what you learned

---

## Key Resources

| Resource | Link |
|---|---|
| YOLOv8 Docs | docs.ultralytics.com |
| Supervision Docs | supervision.roboflow.com |
| CS231n Lectures | youtube.com (search CS231n 2017) |
| BDD100K Dataset | bdd-data.berkeley.edu |
| SORT Paper | arxiv.org/abs/1602.00763 |
