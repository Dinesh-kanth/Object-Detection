from ultralytics import YOLO
import cv2
import math
import pandas as pd
from collections import defaultdict
# Start webcam
webcam = cv2.VideoCapture(0)
webcam.set(3, 640)  # Set width
webcam.set(4, 480)  # Set height
# Load YOLO model
yolo_model = YOLO("yolov8s.pt")  # large model
# Object class names
object_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair dryer",
    "toothbrush"
]
# Track object IDs and positions
object_counter = defaultdict(int)
log_data = []
frame_counter = 0  # Initialize frame counter
while True:
    frame_success, frame = webcam.read()
    if not frame_success:
        break
    frame_counter += 1
    if frame_counter % 3 != 0:  # Process every 3rd frame
        continue
    # Resize frame for faster detection
    small_frame = cv2.resize(frame, (320, 240))  # Downscaling the frame
    detection_results = yolo_model(small_frame, stream=True)
    for detection in detection_results:
        bounding_boxes = detection.boxes
        for bounding_box in bounding_boxes:
            # Bounding box coordinates for the resized frame
            x1, y1, x2, y2 = bounding_box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Adjust bounding box to match original frame size
            x1 = int(x1 * (640 / 320))
            y1 = int(y1 * (480 / 240))
            x2 = int(x2 * (640 / 320))
            y2 = int(y2 * (480 / 240))
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # Confidence score and class name
            confidence_score = math.ceil((bounding_box.conf[0] * 100)) / 100
            class_index = int(bounding_box.cls[0])
            class_name = object_classes[class_index]
            # Track object with ID
            object_counter[class_name] += 1
            object_id = object_counter[class_name]
            # Log detection data
            log_data.append([class_name, confidence_score, x1, y1, x2, y2, object_id])
            # Display object details with unique ID
            text = f"{class_name} {object_id} [{confidence_score}]"
            cv2.putText(frame, text, (x1, y1 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
    # Show frame with bounding boxes and IDs
    cv2.imshow('Webcam', frame)
    # Stop the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break
# Write logged data to CSV file
df = pd.DataFrame(log_data, columns=['Class', 'Confidence', 'x1', 'y1', 'x2', 'y2', 'Object ID'])
df.to_csv("detection_log.csv", index=False)
# Release resources
webcam.release()
cv2.destroyAllWindows()