# Bring in Python's time tools, so we can measure how fast each frame runs
import time

# Bring in OpenCV, the library that talks to the webcam and draws on images
import cv2
# Bring in YOLO, the AI model that finds objects in an image
from ultralytics import YOLO

# Colors in OpenCV are written as (blue, green, red), not (red, green, blue)
# Green color: no blue, full green, no red
GREEN = (0, 255, 0)
# Red color: no blue, no green, full red
RED = (0, 0, 255)
# White color: all three colors at full strength
WHITE = (255, 255, 255)

# Only keep detections the model is at least this sure about (0 = not sure, 1 = fully sure)
MIN_CONFIDENCE = 0.5

# Load the YOLO "small" model (more accurate than "nano", a bit slower)
# The first time, this downloads the model file (yolo11s.pt) automatically
model = YOLO("yolo11s.pt")

# Open the webcam (0 means the first camera on the computer)
cap = cv2.VideoCapture(0)

# Check if the webcam failed to open
if not cap.isOpened():
    # Tell the user something went wrong
    print("Could not open webcam.")
    # Stop the program
    exit()

# Remember the current time, so we can calculate FPS on the first frame
prev_time = time.time()

# Repeat forever, once per video frame, until we break out of the loop
while True:
    # Take one picture (frame) from the webcam; ok tells us if it worked
    ok, frame = cap.read()
    # If taking the picture failed...
    if not ok:
        # ...tell the user...
        print("Could not read frame.")
        # ...and leave the loop
        break

    # Ask YOLO to find objects in this frame, ignoring guesses below MIN_CONFIDENCE
    # verbose=False stops YOLO from printing text for every frame
    # [0] picks the results for our one image
    results = model(frame, conf=MIN_CONFIDENCE, verbose=False)[0]

    # Start the people counter at zero for this frame
    people = 0

    # Go through every object YOLO found, one at a time
    for box in results.boxes:
        # Get the box corners: (x1, y1) = top-left, (x2, y2) = bottom-right, as whole numbers
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Turn the class number (like 0) into a readable name (like "person")
        name = model.names[int(box.cls[0])]
        # Get how sure the model is about this object, as a normal number
        conf = float(box.conf[0])

        # If the object is a person...
        if name == "person":
            # ...use green for its box...
            color = GREEN
            # ...and add one to the people counter
            people += 1
        # Otherwise (any other object)...
        else:
            # ...use red for its box
            color = RED

        # Draw the box around the object, 2 pixels thick
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Build the label text, e.g. "person 0.87" (confidence with 2 decimals)
        label = f"{name} {conf:.2f}"
        # Measure how wide (w) and tall (h) the label text will be
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Draw a filled rectangle above the box as a background for the label (-1 = filled)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
        # Write the label text in white on top of that background
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    # Get the current time
    now = time.time()
    # FPS = 1 divided by how many seconds this frame took
    fps = 1 / (now - prev_time)
    # Save the current time for the next frame's calculation
    prev_time = now

    # Write the FPS in the top-left corner (1 decimal place)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
    # Write the number of people just below the FPS
    cv2.putText(frame, f"People: {people}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)

    # Show the finished frame in a window called "Spotter"
    cv2.imshow("Spotter", frame)

    # Wait 1 millisecond for a key press; if the key is "q", leave the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        # Leave the loop so the program can finish
        break

# Let go of the webcam so other apps can use it
cap.release()
# Close the "Spotter" window
cv2.destroyAllWindows()
