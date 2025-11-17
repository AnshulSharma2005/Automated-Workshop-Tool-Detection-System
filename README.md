Automated Workshop Tool Detection System

A complete end-to-end AI-powered system for detecting workshop tools such as Screwdrivers, Hammers, Pliers, Scissors, Spanners, Files, and Pincers using YOLOv8 and an interactive Streamlit UI.

This project automates dataset cleaning, model training, and real-time inference for workshop tool detection.

🚀 Features

✔ YOLOv8-based custom object detection

✔ Fully cleaned & auto-fixed dataset

✔ Streamlit-based interactive UI

✔ Supports both image & video detection

✔ Class-wise object count visualization

✔ Optimized layout for better UX

✔ 100% test image detection accuracy in final model

📂 Project Structure
Automated-Workshop-Tool-Detection-System/
│── data/
│   ├── ToolsFixed/               # Clean + auto-fixed dataset
│── models/
│── streamlit_app.py              # Streamlit UI
│── fix_dataset.py                # Dataset auto-fixing script
│── test_detection.py             # Quick test script
│── requirements.txt
│── README.md                     # You are reading this file

🛠️ Installation
pip install -r requirements.txt

▶️ Run the Streamlit Web App
streamlit run streamlit_app.py

🧪 Run Detection on a Single Image
yolo predict model=runs/detect/train12/weights/best.pt source="path/to/image.jpg" save=True

🧹 Dataset Auto-Fixing

fix_dataset.py automatically:

Removes corrupt images

Deletes duplicate images

Fixes mismatched label files

Validates YOLO annotation formats

Reorganizes files into:

train/

valid/

test/

Generates updated data.yaml

This produced the final clean dataset: ToolsFixed.

🧠 Training Command
yolo train model=yolo11n.pt data=data/ToolsFixed/data.yaml epochs=50 imgsz=640

📊 Final Model Performance

✔ All 21 test images correctly detected

✔ All 7 classes recognized

✔ Robust predictions on real workshop images

✔ Strong performance even on cluttered backgrounds

🔧 Weekly Progress — Improvements by Me
Week 1

Set up initial project

Integrated YOLOv8

Tested pipeline using sample dataset

Week 2

Cleaned dataset

Identified issues:

duplicate images

mismatched labels

wrong nc value

missing classes

Wrote automatic dataset fixing script (fix_dataset.py)

Week 3

Generated fully cleaned dataset → ToolsFixed

Corrected class names & YAML file

Retrained the model for improved accuracy

Week 4

Improved Streamlit UI:

compact layout

modern styling

better image/video preview

added class-wise detection summary

Completed full testing & final evaluation

👨‍💻 Author

Anshul Sharma
GitHub: https://github.com/AnshulSharma2005

📜 License

This project is open-source and free to use for educational and research purposes.
