import cv2
import time
from ultralytics import YOLO
from datetime import datetime
import winsound
import threading
import os

# Load YOLOv5 model
model = YOLO("yolov5s.pt")

# Allowed classes to detect; set to None to allow all classes
ALLOWED_CLASSES = ["person", "car", "dog"]

# Initial confidence threshold
conf_threshold = 0.5

# Cooldown between detections (in seconds)
BEEP_COOLDOWN = 3
last_beep_time = 0

# Sound file
CUSTOM_SOUND_FILE = "siren.wav"

# Output directory for captured images
CAPTURE_DIR = "captured_people"
os.makedirs(CAPTURE_DIR, exist_ok=True)

def on_trackbar(val):
    global conf_threshold
    conf_threshold = val / 100

def play_sound_for_5_seconds():
    end_time = time.time() + 5
    while time.time() < end_time:
        winsound.PlaySound(CUSTOM_SOUND_FILE, winsound.SND_FILENAME)

def capture_and_save_image(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"[INFO] Image captured and saved as {filename}")

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create window and confidence slider
cv2.namedWindow("Webcam - Object Detection")
cv2.createTrackbar("Confidence", "Webcam - Object Detection", int(conf_threshold * 100), 100, on_trackbar)

# Log file
log_file = open("detections_log.txt", "w")

# FPS timing
prev_time = 0

print("Press 'q' to quit, 's' to save a screenshot manually.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    display_frame = frame.copy()
    current_time = time.time()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]

            if ALLOWED_CLASSES and label not in ALLOWED_CLASSES:
                continue
            if confidence < conf_threshold:
                continue

            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Trigger actions for person detection
            if label == "person" and (current_time - last_beep_time) > BEEP_COOLDOWN:
                threading.Thread(target=play_sound_for_5_seconds, daemon=True).start()
                capture_and_save_image(frame)
                last_beep_time = current_time

            # Log detection
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_file.write(f"[{timestamp}] {label} - {confidence:.2f}\n")

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Display FPS
    cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Webcam - Object Detection", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('s'):  # Save screenshot manually
        screenshot_name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(screenshot_name, display_frame)
        print(f"[INFO] Screenshot saved as {screenshot_name}")

# Cleanup
log_file.close()
cap.release()
cv2.destroyAllWindows()
