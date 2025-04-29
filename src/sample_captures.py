import depthai as dai
import cv2
import os
import time
import numpy as np
from ultralytics import YOLOv10
import supervision as sv

# 📁 Define base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Where this script is located
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')
OUTPUT_DIR = os.path.join(BASE_DIR, 'example_images')
FLIP_IMAGE = True

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv10 model
model = YOLOv10(MODEL_PATH)
bounding_box_annotator = sv.BoxAnnotator()

# Set up pipeline
pipeline = dai.Pipeline()
cam_rgb = pipeline.createColorCamera()
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
cam_rgb.initialControl.setManualFocus(135)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam_rgb.video.link(xout.input)

# Counter
image_count = 1

# Start device
with dai.Device(pipeline) as device:
    q = device.getOutputQueue("video", maxSize=1, blocking=False)

    while True:
        frame = q.get().getCvFrame()
        if FLIP_IMAGE:
            frame = cv2.flip(frame, -1)  # Flip if camera mounted upside down
        frame_resized = cv2.resize(frame, (1280, 720))


        # Run YOLO detection
        results = model(frame_resized, conf=0.5)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Draw detections for visualization
        annotated_frame = bounding_box_annotator.annotate(scene=frame_resized.copy(), detections=detections)
        cv2.imshow("OAK-1 Detection Preview", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(detections.xyxy) > 0:
                filename = os.path.join(OUTPUT_DIR, f"sample_{image_count}.jpg")
                cv2.imwrite(filename, frame)  # Save original (non-annotated) frame
                print(f"✅ Captured with detection: {filename}")
                image_count += 1
            else:
                print("❌ No detection found. Not saving frame.")
        elif key == ord('q'):
            break

cv2.destroyAllWindows()
