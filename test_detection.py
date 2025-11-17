from ultralytics import YOLO
import cv2

# --- Load your trained model ---
model_path = "runs/models/tool_detection/weights/best.pt"
model = YOLO(model_path)

# --- Path to the image you want to test ---
image_path = "data/Tools.v1i.yolov8/train/images/1_jpg.rf.c6c4b644e95b894a371b39f965161763.jpg"

# --- Run prediction ---
results = model(image_path, conf=0.05, imgsz=640)

# --- Display results ---
for r in results:
    print("Detected boxes:", r.boxes)
    if r.boxes is not None and len(r.boxes) > 0:
        print("✅ Object(s) detected!")
    else:
        print("❌ No objects detected.")

    # Save an annotated image
    annotated = r.plot()  # draws boxes
    cv2.imwrite("result.jpg", annotated)
    print("Result saved as result.jpg")
