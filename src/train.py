# src/train.py
"""
Train script for YOLOv8.
Example:
    python src/train.py --data data/data.yaml --model yolov8n.pt --epochs 50 --img 640 --batch 8 --device cpu
"""

import argparse
from ultralytics import YOLO

def main(args):
    model = YOLO(args.model)  # automatically downloads if not present
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        save=True,                       # ✅ always save best.pt & last.pt
        project="runs/models",           # ✅ custom save directory
        name="tool_detection"            # ✅ run name
    )
    print("✅ Training finished. Check runs/models/tool_detection/weights/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/data.yaml', help='path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='yolov8 checkpoint (yolov8n.pt etc.)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--img', type=int, default=640, help='image size')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu', help="'0' for GPU 0, 'cpu' for CPU")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=50)
    args = parser.parse_args()
    main(args)
