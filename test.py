import torch
import cv2
import os

# Load YOLOv5 Model (pre-trained on COCO dataset)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)

# Open Video
video_path = "/home/nithin04ch/testy/Ambulance.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create output directory
output_dir = "/home/nithin04ch/testy/output"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame is read (end of video)

    # Run YOLOv5 model on frame
    results = model(frame)
    detections = results.pandas().xyxy[0]

    ambulance_detected = False
    vehicle_count = 0

    for _, row in detections.iterrows():
        class_name = row["name"]
        confidence = row["confidence"]

        if class_name in ["car", "truck", "bus", "motorcycle"]:
            vehicle_count += 1

        if class_name == "truck" and confidence > 0.6:
            x1, y1, x2, y2 = map(int, [row["xmin"], row["ymin"], row["xmax"], row["ymax"]])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"Ambulance ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ambulance_detected = True

    if ambulance_detected:
        print("🚑 Ambulance detected in lane! Clear the lane.")

    # Display vehicle count on top of frame
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Save frame instead of displaying
    frame_count += 1
    cv2.imwrite(f"{output_dir}/frame_{frame_count}.jpg", frame)

cap.release()
print(f"✅ Processed frames saved in {output_dir}")
