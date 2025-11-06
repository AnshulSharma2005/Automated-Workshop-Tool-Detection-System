# src/detect.py
"""
Run detection with trained weights.
Example:
    python src/detect.py --weights runs/detect/train/weights/best.pt --img data/images/valid/some.jpg --save out.jpg
"""

import argparse
import cv2
from ultralytics import YOLO
import os

def main(args):
    model = YOLO(args.weights)
    results = model(args.img, imgsz=args.imgsz)  # list-like
    res0 = results[0]
    out = res0.plot()
    if args.save:
        cv2.imwrite(args.save, out)
        print(f"Saved {args.save}")
    else:
        cv2.imshow("Detection", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/detect/train/weights/best.pt')
    parser.add_argument('--img', type=str, required=True, help='path to input image')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--save', type=str, default=None, help='path to save output image')
    args = parser.parse_args()
    main(args)
