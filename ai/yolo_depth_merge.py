from google.colab import drive
import os

drive.mount('/content/drive')
base_path = '/content/drive/MyDrive/DL_Project'

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import cv2

from ultralytics import YOLO

import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
import glob

import json
from datetime import datetime
from scripts.atomic_write import atomic_write

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

  def forward(self, x):
      return self.conv(x)

class PotholeUNet(nn.Module):
    def __init__(self):
        super(PotholeUNet, self).__init__()

        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2) 

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2) 

        self.down3 = DoubleConv(128, 256)  
        self.pool3 = nn.MaxPool2d(2)      

        self.down4 = DoubleConv(256, 512)  
        self.pool4 = nn.MaxPool2d(2)      

        self.bottleneck = DoubleConv(512, 1024) 

        self.upConv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up4 = DoubleConv(1024, 512)

        self.upConv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = DoubleConv(512, 256)

        self.upConv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = DoubleConv(256, 128)

        self.upConv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        p1 = self.pool1(x1)

        x2 = self.down2(p1)
        p2 = self.pool2(x2)

        x3 = self.down3(p2)
        p3 = self.pool3(x3)

        x4 = self.down4(p3)
        p4 = self.pool4(x4)

        b = self.bottleneck(p4)

        u4 = self.upConv4(b)
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.up4(u4)

        u3 = self.upConv3(u4)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.up3(u3)

        u2 = self.upConv2(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up2(u2)

        u1 = self.upConv1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.up1(u1)

        out = self.out(u1)
        return torch.sigmoid(out)

def calculate_real_depth(d_road, pothole_region, max_depth_cm=20.0):

    if pothole_region.size == 0:
        return 0.0

    d_bottom = np.percentile(pothole_region, 1)

    pixel_diff = abs(d_road - d_bottom)

    actual_depth_cm = pixel_diff * max_depth_cm

    actual_depth_m = actual_depth_cm / 100

    return actual_depth_m

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#원하는 사진 경로 입력
target_image = ' '
input_image = Image.open(target_image).convert('RGB')
input_tensor = preprocess(input_image).unsqueeze(0).to(device)

model = PotholeUNet().to(device)
model.eval()

ensemble_pred = torch.zeros((1, 1, 256, 256)).to(device)

with torch.no_grad():
    for fold in range(1, 6):
        #현재 파일 기준 상대 경로 입력 (밖에 있다고 가정)
        load_path = f'../best_model/best_model_fold{fold}.pth'
        model.load_state_dict(torch.load(load_path))

        output = model(input_tensor)

        ensemble_pred += output

ensemble_pred = ensemble_pred / 5.0
pred_map = ensemble_pred.squeeze().cpu().numpy()

#현재 사진 기준 앞 사다리꼴 모양이 도로라고 가정 후 해당 사다리꼴 내에서의 중앙값이 정상 도로의 픽셀이라고 결정
mask = Image.new('L', (256, 256), 0)
draw = ImageDraw.Draw(mask)

top_left = (int(256 * 0.4), int(256 * 0.3))
top_right = (int(256 * 0.6), int(256 * 0.3))
bottom_left = (0, 256)
bottom_right = (256, 256)
draw.polygon([top_left, top_right, bottom_right, bottom_left], fill=255)

mask_np = np.array(mask)
roi_pixels = pred_map[mask_np == 255]
d_road = np.median(roi_pixels) if roi_pixels.size > 0 else 0.0

yolo_model = YOLO('../weights/best.pt')
detections = yolo_model(target_image, conf=0.25, imgsz=640, iou=0.7, verbose=False)

orig_w, orig_h = input_image.size
pred_map_resized = cv2.resize(pred_map, (orig_w, orig_h))

final_reports = []

now = datetime.now()
date_str = now.strftime("%Y%m%d")
iso_time_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
photo_lat = float(os.getenv("PHOTO_LAT", "37.551302"))
photo_lng = float(os.getenv("PHOTO_LNG", "127.075108"))

#YOLO 모델에서 찾은 파손된 도로들의 위치
for result in detections:
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy() 
    class_ids = result.boxes.cls.cpu().numpy()
    class_names_dict = yolo_model.names 

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        conf = float(confs[idx]) 
        
        cls_id = int(class_ids[idx])
        raw_type_name = class_names_dict[cls_id] 
        
        #파손 도로의 명칭
        if raw_type_name.lower() == 'pothole':
            final_type_name = 'pothole'
            prefix = 'ph'
        else:
            final_type_name = 'crack'
            prefix = 'cr'
        
        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)

        #파손 도로의 위치만 보도록 설정
        pothole_region = pred_map_resized[y:y+h, x:x+w]

        #파손 도로의 깊이 계산
        depth_m = calculate_real_depth(d_road, pothole_region)

        pothole_id = f"{prefix}-{date_str}-{str(len(final_reports) + 1).zfill(3)}"

        report_dict = {
            "id": pothole_id,
            "type": final_type_name,
            "lat": photo_lat, 
            "lng": photo_lng,
            "width_m": w,       
            "length_m": h,      
            "depth_m": round(depth_m, 3),
            "confidence": round(conf, 2), 
            "detected_at": iso_time_str,
        }
        
        final_reports.append(report_dict)

print("최종 파손 도로 탐지 & 깊이")
for report in final_reports:
    print(report)

# atomic write
# 저장할 파일 경로
output_file = '../client_3d/result.json'

# 1. 기존 데이터 읽어오기 
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            # 파일이 비어있거나 깨졌을 경우 빈 리스트로 시작
            existing_data = []
else:
    # 파일이 아예 없으면 빈 리스트로 시작
    existing_data = []

existing_data.extend(final_reports)
out_json = json.dumps(existing_data, ensure_ascii=False, indent=4)
atomic_write(output_file, out_json)
