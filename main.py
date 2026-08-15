import argparse
import cv2
from modules.detector        import Detector
from modules.tracker         import VehicleTracker as DeepTracker
from modules.reid_matcher    import ReIDMatcher
from modules.visualizer      import Visualizer
from modules.flow            import SpeedEstimator
from modules.vehicle_tracker import VehicleTracker
parser = argparse.ArgumentParser(description="Rash driver detection pipeline")
parser.add_argument("--input",  required=True,          help="Path to input dashcam video")
parser.add_argument("--output", default="tracked.mp4",  help="Path to save output video (default: tracked.mp4)")
args = parser.parse_args()

cap    = cv2.VideoCapture(args.input)
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {width}x{height}, {fps:.2f} fps")

detector      = Detector("yolov8s.pt", conf=0.35, frame_w=width, frame_h=height)
tracker       = DeepTracker()
matcher       = ReIDMatcher(tracker._tracker.model, frame_w=width, frame_h=height)
visualizer    = Visualizer(roi=detector.roi)
estimator     = SpeedEstimator(fps=fps)
vehicle_hist  = VehicleTracker()

writer     = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
frame_idx  = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    dets   = detector.detect(frame)
    tracks = tracker.update(dets, frame)
    tracks = matcher.update(tracks, frame)
    tracks = estimator.update(frame, tracks)
    vehicle_hist.update(width, height, tracks)

    # print feature summary every 30 frames
    if frame_idx % 30 == 0:
        for track in tracks:
            tid = track[4]
            feats = vehicle_hist.get_features(tid)
            if feats:
                print(f"[frame {frame_idx}] ID:{tid} | "
                      f"avg_spd={feats['avg_speed']:.1f} "
                      f"max_spd={feats['max_speed']:.1f} "
                      f"var={feats['speed_variance']:.1f} "
                      f"brakes={feats['braking_count']} "
                      f"weaves={feats['weave_count']}")

    visualizer.draw(frame, tracks)
    writer.write(frame)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

writer.release()
cap.release()
cv2.destroyAllWindows()
print(f"Saved to {args.output}")
