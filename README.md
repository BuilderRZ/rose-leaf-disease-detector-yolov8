# 🌹 Rose Leaf Disease Detector using YOLOv8

A smart and lightweight AI web application to detect the health 
status of rose leaves using YOLOv8. This system was trained on a custom-labeled 
dataset using Roboflow and includes a CNN-based pre-filter to ensure only rose leaves are analyzed. 
The app is built with Flask and offers an interactive visual interface.

---

## 🚀 Features

- 🧠 **YOLOv8-based disease detection**
- 🌿 **Pre-classification to detect whether the image is a rose leaf or not**
- 📦 **3 disease classes:** Healthy, Mild Infection, Severe Infection
- 📸 **Visual detection with bounding boxes**
- 📊 **Chart-based summary of results**
- 🧰 **Roboflow labeling & YOLOv8 training pipeline**
- 💡 Easy to run locally with simple Flask setup

---

## 🛠️ Tech Stack

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)**
- **Python 3.10+**
- **Flask (for web interface)**
- **Roboflow (dataset labeling & export)**
- **CNN (for rose leaf verification)**

---

## 📷 Detection Example

![Detection Example](static/sample_output.png)

---

## 📁 Project Structure

rose-leaf-disease-detector-yolov8/
├── app.py
├── templates/
│ └── index.html
├── static/
│ └── uploaded/
│ └── sample_output.png
├── yolov8_model/
│ ├── best.pt
│ └── data.yaml
├── utils/
│ └── yolo_inference.py
├── cnn_leaf_classifier/
│ ├── model_cnn.h5
│ └── classify_leaf.py
├── requirements.txt
└── README.md

---
### 📽 Demo Video

<video width="600" controls>
  <source src="assets/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

👉 [Video Demo](https://github.com/BuilderRZ/rose-leaf-disease-detector-yolov8/raw/main/assets/demo.mp4)


## 🚦 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/rose-leaf-disease-detector-yolov8.git
cd rose-leaf-disease-detector-yolov8
Install dependencies:
pip install -r requirements.txt
Run the app:
python app.py
Open in browser:
http://127.0.0.1:5000
📌 Notes
The YOLOv8 model is located in yolov8_model/best.pt and trained on rose leaf datasets annotated with Roboflow.

All uploaded images are processed and visualized with bounding boxes + disease info.

The CNN model helps filter out non-rose leaf images before running YOLOv8.


📜 License
MIT License © 2025 Syaiful Rizal / Nebulir Labs

🤖 Credits
Dataset labeled using Roboflow

YOLOv8 by Ultralytics

Developed with ❤️ by Rizal / Nebulir Labs


