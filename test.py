import torch
import cv2

# Load YOLOv5 Model (pre-trained on COCO dataset)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)  # Use 'yolov5m' or 'yolov5l' for better accuracy

# Open Video
video_path = "/home/nithin04ch/testy/Ambulance.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame is read (end of video)

    # Run YOLOv5 model on frame (OpenCV BGR image)
    results = model(frame)

    # Extract detections
    detections = results.pandas().xyxy[0]
    ambulance_detected = False  # Flag to check if ambulance is detected
    vehicle_count = 0  # Counter for vehicles

    for _, row in detections.iterrows():
        class_name = row["name"]
        confidence = row["confidence"]

        # Count vehicles (YOLO detects them separately)
        if class_name in ["car", "truck", "bus", "motorcycle"]:
            vehicle_count += 1  # Increment vehicle count

        # Detect ambulances (YOLO might classify them as trucks)
        if class_name == "truck" and confidence > 0.6:
            x1, y1, x2, y2 = map(int, [row["xmin"], row["ymin"], row["xmax"], row["ymax"]])

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"Ambulance ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ambulance_detected = True

    if ambulance_detected:
        print("🚑 Ambulance detected in lane! Clear the lane.")

    # Display vehicle count on top of frame
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Display frame
    cv2.imshow("Ambulance Detection & Vehicle Count", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
