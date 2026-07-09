import cv2
# cv2 is a toolbox to work with images frame by frame
import numpy as np

# now importing YOLO for detection:
from ultralytics import YOLO
capture = cv2.VideoCapture(0)

 #Load a pre-trained model (nano version is fast and lightweight)
model = YOLO("yolov8n.pt")  
from sort import Sort
tracker = Sort()  # this keeps track of object IDs across frames
def auto_detection():

    while True:
        ret, frame = capture.read()
        if not ret:
            break
           # Run object detection on an image
        results = model(frame)
          # Draw detections on frame
        annotated_frame = results[0].plot()

        cv2.imshow("capture from window with YOLO", annotated_frame)

        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()



def manual_detection():
   
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)
        boxes = results[0].boxes

        detections = []  # this will store filtered detections for SORT

        # ------------------ STEP 1: FILTER + PREPARE DATA ------------------
        for box in boxes:
            cls = int(box.cls[0])        # get class ID
            conf = float(box.conf[0])   # get confidence score

            #  FILTER: keep only PERSON (class 0)
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Add detection in SORT format: [x1, y1, x2, y2, confidence]
                detections.append([x1, y1, x2, y2, conf])

        # ------------------ STEP 2: CONVERT TO NUMPY ------------------
        if len(detections) == 0:
            detections = np.empty((0, 5))  # avoid crash if no detections
        else:
            detections = np.array(detections)

        # ------------------ STEP 3: TRACK OBJECTS ------------------
        tracked_objects = tracker.update(detections)

        # ------------------ STEP 4: DRAW TRACKED OBJECTS ------------------
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw ID
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show result
        cv2.imshow("Manual YOLO + Tracking", frame)

        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


# ------------------ CHOOSE WHICH ONE TO RUN 
# AUTO MODE:
# Uses YOLO built-in visualization

# MANUAL MODE:
# Custom pipeline:
# 1. Filter detections
# 2. Convert format
# 3. Apply SORT tracking
# 4. Draw boxes + IDs manually

mode = "manual"  

if mode == "auto":
    auto_detection()
else:
    manual_detection()






