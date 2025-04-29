import cv2
import os
import numpy as np
from ultralytics import YOLOv10
import supervision as sv

# Define base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Where this script is located
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')
EXAMPLES_DIR = os.path.join(BASE_DIR, 'example_images')

# Load YOLOv10 model
model = YOLOv10(MODEL_PATH)
bounding_box_annotator = sv.BoxAnnotator()

#  Get list of images
image_files = [f for f in os.listdir(EXAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print(" No images found in example_images/")
    exit()

# Process each image
for img_name in image_files:
    img_path = os.path.join(EXAMPLES_DIR, img_name)
    image = cv2.imread(img_path)

    if image is None:
        print(f" Could not load image: {img_name}")
        continue

    # 🔎 Run detection
    results = model(image, conf=0.5)[0]
    detections = sv.Detections.from_ultralytics(results)

    if len(detections.xyxy) > 0:
        print(f"✅ Detected objects in: {img_name}")
        #  Draw detections
        annotated_image = bounding_box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = cv2.resize(annotated_image, (1280, 720))
        cv2.imshow(f"Detection: {img_name}", annotated_image)
        cv2.waitKey(0)  # Wait for key press to move to next
        cv2.destroyWindow(f"Detection: {img_name}")
    else:
        print(f"❌ No objects detected in: {img_name}")

print("✅ Finished processing all images.")
