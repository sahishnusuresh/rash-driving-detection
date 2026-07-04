from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
print(model.names)
cap=cv2.VideoCapture('dashcam.mp4')
fps    = cap.get(cv2.CAP_PROP_FPS)           # frames per second
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {width}x{height}, {fps} fps")

writer = cv2.VideoWriter('detected.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # frame is a NumPy array of shape (height, width, 3) — BGR color
    results = model(frame, classes=[2, 3, 5, 7])
    r = results[0]

    annotated = r.plot()
    writer.write(annotated)
    cv2.imshow("Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

writer.release()
cap.release()
cv2.destroyAllWindows()
print("Saved to detected.mp4")