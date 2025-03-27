from flask import Flask, Response, render_template
import cv2
import torch

app = Flask(__name__)

# Load YOLOv5 Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)

# Open Video Capture (Change to 0 for webcam or use video file)
video_path = "/home/nithin04ch/testy/Ambulance.mp4"
cap = cv2.VideoCapture(video_path)

def detect_objects():
    """ Generator function to stream video with object detection """
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if video ends

        # Run YOLOv5 on the frame
        results = model(frame)
        detections = results.pandas().xyxy[0]

        ambulance_detected = False  # Flag for ambulance detection
        vehicle_count = 0  # Count vehicles

        for _, row in detections.iterrows():
            class_name = row["name"]
            confidence = row["confidence"]

            # Count vehicles
            if class_name in ["car", "truck", "bus", "motorcycle"]:
                vehicle_count += 1

            # Detect Ambulances (Assumed as trucks)
            if class_name == "truck" and confidence > 0.6:
                x1, y1, x2, y2 = map(int, [row["xmin"], row["ymin"], row["xmax"], row["ymax"]])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"Ambulance ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                ambulance_detected = True

        # Display vehicle count
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Convert frame to JPEG format
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/")
def index():
    """ Home Page """
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """ Video Streaming Route """
    return Response(detect_objects(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
