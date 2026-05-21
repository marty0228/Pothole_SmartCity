"""
Lightweight, local-friendly version of Yolo + depth merge.
Usage:
    python ai/yolo_depth_merge.py --image path/to/image.jpg --weights weights/best.pt --output client_3d/result.json
"""
import argparse
import os
import json
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True)
parser.add_argument('--weights', default='weights/best.pt')
parser.add_argument('--output', default='client_3d/result.json')
args = parser.parse_args()

# simple helper: dummy depth predictor - replace with real model if available

def calculate_real_depth(d_road, pothole_region, max_depth_cm=20.0):
    if pothole_region.size == 0:
        return 0.0
    d_bottom = np.percentile(pothole_region, 1)
    pixel_diff = abs(d_road - d_bottom)
    actual_depth_cm = pixel_diff * max_depth_cm
    actual_depth_m = actual_depth_cm / 100
    return actual_depth_m

# load image
input_image = Image.open(args.image).convert('RGB')
orig_w, orig_h = input_image.size
input_np = np.array(input_image)

# load yolo
if not os.path.exists(args.weights):
    raise SystemExit(f"Weights not found: {args.weights}. Use scripts/download_weights.ps1 to fetch.")

yolo_model = YOLO(args.weights)
detections = yolo_model(args.image, conf=0.25, imgsz=640, iou=0.7, verbose=False)

# dummy pred_map: use simple gray-scale from image brightness as placeholder
gray = cv2.cvtColor(input_np, cv2.COLOR_RGB2GRAY) / 255.0
pred_map_resized = cv2.resize(gray, (orig_w, orig_h))

# road estimate via lower triangle mask
mask = np.zeros((256,256), dtype=np.uint8)
import numpy as _np
mask_img = Image.new('L', (256,256), 0)
from PIL import ImageDraw
draw = ImageDraw.Draw(mask_img)
draw.polygon([(int(256*0.4), int(256*0.3)), (int(256*0.6), int(256*0.3)), (256,256), (0,256)], fill=255)
mask_np = np.array(mask_img)
roi_pixels = pred_map_resized[mask_np == 255]
d_road = np.median(roi_pixels) if roi_pixels.size > 0 else 0.0

final_reports = []

for result in detections:
    boxes = result.boxes.xyxy.cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2 = box
        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)

        pothole_region = pred_map_resized[y:y+h, x:x+w]
        depth_m = calculate_real_depth(d_road, pothole_region)

        final_reports.append({
            "id": f"ph-{len(final_reports)+1:03d}",
            "type": "pothole",
            "lat": 0.0,
            "lng": 0.0,
            "width_m": float(w),
            "length_m": float(h),
            "depth_m": round(float(depth_m), 3),
            "confidence": float(0.5),
            "detected_at": "now"
        })

# atomic write
from scripts.atomic_write import atomic_write
out_json = json.dumps(final_reports, ensure_ascii=False, indent=2)
atomic_write(args.output, out_json)
print(f"Wrote {len(final_reports)} detections to {args.output}")
