# Object Detection Pipeline with YOLOv10 and OAK-1 Camera

> Adapted and extended from initial YOLOv10 template provided by Roboflow/Nicolai Nielsen to work with Luxonis OAK-1 Camera using DepthAI API.

---

## 📋 Description

This project implements a real-time object detection system using **YOLOv10** models and a **Luxonis OAK-1** camera. It adapts the standard YOLOv10 detection pipeline to integrate **high-resolution video streams**, **manual camera control**, and **bounding box center extraction** for further depth estimation or robotics applications.

Key extensions include:
- Adaptation from standard webcam input to **OAK-1 camera streams** using **DepthAI API**.
- Manual focus setting and 4K resolution capture.
- Real-time bounding box annotation with **Supervision** library.
- Extraction of **bounding box centers** on-demand.
- Efficient resource management: heavy computations are triggered manually.

This pipeline can be used in applications such as:
- Object detection and tracking.
- Depth estimation (bounding box centroids for depth lookup).
- Robotics and automation tasks requiring real-time spatial awareness.

---

## 📂 Project Structure

```
📁 project-root/
├── 📂 src/                # Source code
│    └── detection_pipeline.py
├── 📂 models/             # Trained YOLOv10 models (.pt files)
├── 📂 data/               # Captured images
│    └── detected_objects/
├── 📂 outputs/            # Annotated outputs (optional)
├── 📂 docs/               # Technical documentation (optional)
├── 📄 README.md           # Project description
├── 📄 requirements.txt    # Python dependencies
├── 📄 LICENSE             # License file (MIT)
└── 📄 .gitignore          # Files to ignore (optional)
```

---

## ⚙️ Installation Guide

### 1. Clone and Install YOLOv10 Manually

```bash
git clone https://github.com/THU-MIG/yolov10.git
cd yolov10
git checkout cd2f79c70299c9041fb6d19617ef1296f47575b1
pip install .
```

### 2. Install Required Python Packages

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch Manually

Due to specific CUDA compatibility, install PyTorch manually:

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

---

## 🛠 Technologies Used

- [YOLOv10 (Ultralytics framework)](https://github.com/THU-MIG/yolov10)
- [DepthAI SDK](https://docs.luxonis.com/projects/api/en/latest/)
- [Supervision (annotation library)](https://github.com/roboflow/supervision)
- [OpenCV](https://opencv.org/)
- [HuggingFace Hub](https://huggingface.co/docs/huggingface_hub)
- Python 3.9+

---

## 📚 Background and Context

This work builds upon the concepts developed in my undergraduate thesis:

**"Detección de Objetos y Estimación de Distancia mediante YOLOv10 y Modelos de Profundidad"**, Universidad Simón Bolívar (2025).

In Chapter 4 of the thesis:
- YOLO models were analyzed theoretically and practically.
- Custom training was conducted using **Roboflow**.
- Adaptation to real-time pipelines with custom hardware (OAK-1) was proposed and implemented.

Special extensions implemented in this project:
- Extracting pixel coordinates of bounding box centers for object-centric depth estimation.
- Handling OAK-1 camera streams, including manual autofocus control.
- **Resource-efficient pipeline:** heavy computations (like center and coordinate extraction) are performed only when the user requests it by pressing the `k` key, optimizing real-time performance.

---

## 🎮 Key Functions in Live Stream

- **Real-time Display:**
  - Bounding boxes are shown live with minimal overhead.
- **Capture Image (`c` key):**
  - Saves the current frame without additional heavy annotations.
- **On-demand Center Calculation (`k` key):**
  - Calculates and displays bounding box centers and coordinates only when required.
- **Exit Stream (`q` key):**
  - Closes the video stream and windows.

This strategy ensures an optimized, responsive application suitable for real-world deployments.

---

## 📸 Sample Workflow

1. Initialize OAK-1 camera pipeline with 4K resolution.
2. Load a trained **YOLOv10** model (`best.pt`).
3. Capture live frames from the OAK-1.
4. Run YOLOv10 inference on each frame.
5. Annotate detections (bounding boxes, labels).
6. Calculate and display bounding box centers on-demand.
7. Save frames and detection data via user input.

---

## ⚠️ Known Issues and Fixes

### 1. OpenCV GUI Errors (`cv2.imshow`, `cv2.FONT_HERSHEY_SIMPLEX`)

**Problem:**  
You might encounter errors like:

- `cv2.error: The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support`
- `AttributeError: module 'cv2' has no attribute 'FONT_HERSHEY_SIMPLEX'`

**Cause:**  
The `supervision` package may automatically install `opencv-python-headless`, which lacks GUI support necessary for displaying images and rendering text.

**Solution:**  
You must ensure that only `opencv-python` is installed, not `opencv-python-headless`.

1. Uninstall both OpenCV versions:

```bash
pip uninstall opencv-python
pip uninstall opencv-python-headless
```

2. Reinstall the correct OpenCV version:

```bash
pip install opencv-python==4.8.0.76
```

3. Verify installation:

```bash
pip list
```
Make sure only `opencv-python` appears, and `opencv-python-headless` is NOT listed.

---

## 🤝 Credits

- Based on the initial **YOLOv10** training template by **Roboflow** and **Nicolai Nielsen**.
- Extended and adapted by **Rolando Cortez García** for use with DepthAI-powered Luxonis OAK-1 camera systems.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 📬 Contact

**Rolando Cortez García**  
Email: rolscg@gmail.com  
LinkedIn: [https://www.linkedin.com/in/rolando-cortez/](https://www.linkedin.com/in/rolando-cortez/)  
GitHub: [https://github.com/rolandocortez](https://github.com/rolandocortez)

Always open to collaborations, freelance opportunities, and computer vision challenges!

---

> "Detect, analyze, adapt — one frame at a time." 🚀
