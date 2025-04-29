import cv2
import supervision as sv
from ultralytics import YOLOv10
import depthai as dai
import os
import time

# Load YOLO model
model = YOLOv10(r'yolov10-oak1-bbox-center-extraction/models/best.pt')

# Initialize annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Create a directory to store captured images
output_dir = "yolov10-oak1-bbox-center-extraction/detected_objects"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

# Counter for captured images
image_count = 0

# Naming format for images
prefix = "image_"
file_format = ".jpg"  # You can change to .png if preferred

# Function to calculate the center of a bounding box
def calculate_center(x_min, y_min, x_max, y_max):
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return center_x, center_y

flag_corrected_image = False

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
        # Flip the image because the camera in the experiment was upside down
        annotated_image = cv2.flip(annotated_image, -1)
        # Show the annotated image
        cv2.imshow("Capture", annotated_image)

        # Check if the 'c' key is pressed to capture the image
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Filename based on counter and timestamp to avoid duplicates
            filename = os.path.join(output_dir, f"{prefix}{image_count}_{int(time.time())}{file_format}")
            
            # Save the image
            cv2.imwrite(filename, frame)
            print(f"Image saved: {filename}")
            image_count += 1

        # Press 'q' to exit the loop
        if key == ord('q'):
            break

        # Press 'k' to check detections and save the last frame with centers
        if key == ord('k'):
            print(f'detections: {detections}')
            flag_corrected_image = False

            # Check if there are detections
            if len(detections.xyxy) > 0:
                # Save the last frame with detections
                last_detection_frame = frame.copy()
                coord_text_cache = []
                # Draw coordinates and calculate the center of the bounding box
                for i, bbox in enumerate(detections.xyxy):
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    center_x, center_y = calculate_center(x_min, y_min, x_max, y_max)

                    # Draw the bounding box and center point
                    cv2.rectangle(last_detection_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    cv2.circle(last_detection_frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
                    
                    # Write coordinates on the top-left corner
                    coord_text = f"BBox: ({x_min}, {y_min}, {x_max}, {y_max}), Center: ({int(center_x)}, {int(center_y)})"
                    coord_text_cache.append(coord_text)
                
                last_detection_frame = cv2.resize(last_detection_frame, (1280, 680))
                # Also flip the image with drawn centers
                last_detection_frame = cv2.flip(last_detection_frame, -1)

                for i in range(len(detections.xyxy)):
                    cv2.putText(last_detection_frame, coord_text_cache[i], (10, 30 * (i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Show the last frame with detections and coordinates
                cv2.imshow("Last Detection Frame", last_detection_frame)
                print(f'bbox: {detections.xyxy}')
                cv2.waitKey(0)  # Wait for user input to close the window

# Close all OpenCV windows
cv2.destroyAllWindows()