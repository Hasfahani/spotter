import time

import cv2
from ultralytics import YOLO

# Colors in OpenCV are BGR (blue, green, red), not RGB
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

# Load the YOLO nano model (downloads the .pt file the first time)
model = YOLO("yolo11n.pt")

# Open the default webcam (0 = first camera on the system)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

prev_time = time.time()

while True:
    # Grab one frame from the camera
    ok, frame = cap.read()
    if not ok:
        print("Could not read frame.")
        break

    # Run detection on the frame; [0] = results for our single image
    results = model(frame, verbose=False)[0]

    people = 0

    for box in results.boxes:
        # Box corners, class name and confidence
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        # Green for people, red for everything else
        if name == "person":
            color = GREEN
            people += 1
        else:
            color = RED

        # Draw the box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw the label on a filled background so it's readable
        label = f"{name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    # FPS = 1 / time it took to process this frame
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    # Show FPS and people count in the top-left corner
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
    cv2.putText(frame, f"People: {people}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)

    # Show the frame in a window called "Spotter"
    cv2.imshow("Spotter", frame)

    # Wait 1 ms for a key press; quit if it's "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
