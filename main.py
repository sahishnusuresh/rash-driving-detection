import argparse
import cv2
from modules.detector     import Detector
from modules.tracker      import VehicleTracker
from modules.reid_matcher import ReIDMatcher
from modules.visualizer   import Visualizer

parser = argparse.ArgumentParser(description="Rash driver detection pipeline")
parser.add_argument("--input",  required=True,          help="Path to input dashcam video")
parser.add_argument("--output", default="tracked.mp4",  help="Path to save output video (default: tracked.mp4)")
args = parser.parse_args()

cap    = cv2.VideoCapture(args.input)
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {width}x{height}, {fps:.2f} fps")

detector   = Detector("yolov8s.pt", conf=0.35, frame_w=width, frame_h=height)
tracker    = VehicleTracker()
matcher    = ReIDMatcher(tracker._tracker.model, frame_w=width, frame_h=height)
visualizer = Visualizer(roi=detector.roi)

writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    dets   = detector.detect(frame)
    tracks = tracker.update(dets, frame)
    tracks = matcher.update(tracks, frame)

    visualizer.draw(frame, tracks)
    writer.write(frame)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

writer.release()
cap.release()
cv2.destroyAllWindows()
print(f"Saved to {args.output}")
