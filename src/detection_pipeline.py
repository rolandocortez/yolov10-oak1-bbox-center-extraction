import cv2
import supervision as sv
from ultralytics import YOLOv10
import depthai as dai
import os
import time
import numpy as np

# Whether to flip the camera image vertically and horizontally (for custom setups)
FLIP_IMAGE = True

# Define dynamic paths based on project structure
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
CAPTURES_DIR = os.path.join(OUTPUT_DIR, 'captures')
DETECTED_DIR = os.path.join(OUTPUT_DIR, 'detected_centers')

# Create necessary directories if they don't exist
for folder in [OUTPUT_DIR, CAPTURES_DIR, DETECTED_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load YOLO model
model = YOLOv10(MODEL_PATH)

# Initialize annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Initialize the OAK-1 camera with DepthAI
pipeline = dai.Pipeline()

# Configure the RGB camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
cam_rgb.initialControl.setManualFocus(135)

# Configure the video stream output
xout_video = pipeline.create(dai.node.XLinkOut)
xout_video.setStreamName("isp")
cam_rgb.video.link(xout_video.input)

# Counters for captured and detected images
capture_counter = 1
detection_counter = 1

# Naming format for images
prefix_capture = "capture_"
prefix_detected = "detected_center_"
file_format = ".jpg"  # You can change to .png if preferred

# Function to calculate the center of a bounding box
def calculate_center(x_min, y_min, x_max, y_max):
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return center_x, center_y

flag_corrected_image = False
showing_detection_window = False
in_capture_mode = True
waiting_for_detection = False
waiting_window_open = False
detection_frame_streak = 0  # Counter for consecutive detections
DETECTION_FRAMES_REQUIRED = 3  # How many frames of detection are needed

# Execute the pipeline and process video in real-time
with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="isp", maxSize=1, blocking=True)

    while True:
        # Capture frame from the OAK camera
        in_video = video_queue.get()
        frame = in_video.getCvFrame()

        # Annotate detections on the image (without saving yet)
        results = model(frame, conf=0.5)[0]
        detections = sv.Detections.from_ultralytics(results)
        annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        # Resize the image for better display on the monitor
        annotated_image = cv2.resize(annotated_image, (1280, 720))
        if FLIP_IMAGE:
            annotated_image = cv2.flip(annotated_image, -1)

        # Show the annotated image
        cv2.imshow("Capture", annotated_image)

        # Show waiting window if needed
        if waiting_for_detection and not waiting_window_open:
            waiting_image = 255 * np.ones((400, 600, 3), dtype=np.uint8)
            cv2.putText(waiting_image, "Waiting for detection...", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(waiting_image, "Press 'r' to return.", (90, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Waiting", waiting_image)
            waiting_window_open = True

        key = cv2.waitKey(1) & 0xFF

        if waiting_for_detection:
            if len(detections.xyxy) > 0:
                detection_frame_streak += 1
            else:
                detection_frame_streak = 0

            if detection_frame_streak >= DETECTION_FRAMES_REQUIRED:
                # Save detected frame
                last_detection_frame = frame.copy()
                coord_text_cache = []
                for i, bbox in enumerate(detections.xyxy):
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    center_x, center_y = calculate_center(x_min, y_min, x_max, y_max)
                    cv2.rectangle(last_detection_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    cv2.circle(last_detection_frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
                    coord_text = f"BBox: ({x_min}, {y_min}, {x_max}, {y_max}), Center: ({int(center_x)}, {int(center_y)})"
                    coord_text_cache.append(coord_text)

                last_detection_frame = cv2.resize(last_detection_frame, (1280, 680))
                if FLIP_IMAGE:
                    last_detection_frame = cv2.flip(last_detection_frame, -1)

                for i in range(len(detections.xyxy)):
                    cv2.putText(last_detection_frame, coord_text_cache[i], (10, 30 * (i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                filename = os.path.join(DETECTED_DIR, f"{prefix_detected}{detection_counter}{file_format}")
                cv2.imwrite(filename, last_detection_frame)
                print(f"Detection image saved: {filename}")

                cv2.imshow("Last Detection Frame", last_detection_frame)
                showing_detection_window = True
                detection_counter += 1
                waiting_for_detection = False
                detection_frame_streak = 0

                if waiting_window_open:
                    cv2.destroyWindow("Waiting")
                    waiting_window_open = False
                continue

        # Press 'r' to return to capture mode
        if key == ord('r'):
            if showing_detection_window:
                cv2.destroyWindow("Last Detection Frame")
                showing_detection_window = False
            if waiting_window_open:
                cv2.destroyWindow("Waiting")
                waiting_window_open = False
            in_capture_mode = True
            waiting_for_detection = False
            detection_frame_streak = 0

        # Capture image only in capture mode
        if key == ord('c') and in_capture_mode:
            filename = os.path.join(CAPTURES_DIR, f"{prefix_capture}{capture_counter}{file_format}")
            save_frame = frame
            if FLIP_IMAGE:
                save_frame = cv2.flip(save_frame, -1)
            cv2.imwrite(filename, save_frame)
            print(f"Image saved: {filename}")
            capture_counter += 1

        # Quit application with 'q' key
        if key == ord('q'):
            break

        # Detect and prepare to save annotated image with 'k' key
        if key == ord('k') and not waiting_for_detection:
            print("Waiting for the detection of the object to save image. Press 'r' to return to capture mode.")
            in_capture_mode = False
            waiting_for_detection = True
            waiting_window_open = False
            detection_frame_streak = 0

# Close all OpenCV windows
cv2.destroyAllWindows()
