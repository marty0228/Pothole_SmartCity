from google.colab import drive
drive.mount('/content/drive')

import os
import torch
import torch.nn as nn
import numpy as np
import cv2

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ultralytics import YOLO
from IPython.display import display


# 분석할 이미지 경로
TARGET_IMAGE = " "

# best_model_fold1.pth ~ best_model_fold5.pth가 있는 폴더
MODEL_DIR = " "

# YOLO 가중치 경로
YOLO_WEIGHTS_PATH = " "

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



def normalize_depth_map(depth_map):
    depth = depth_map.astype(np.float32)

    min_v = depth.min()
    max_v = depth.max()

    if max_v - min_v < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)

    return (depth - min_v) / (max_v - min_v + 1e-8)


def calculate_real_depth(d_road, pothole_region, max_depth_cm=20.0):
    """
    U-Net depth map 기반 상대 깊이 추정.
    max_depth_cm=20이면 depth 차이 1.0을 20cm로 환산.
    """

    if pothole_region is None or pothole_region.size == 0:
        return 0.0

    d_bottom = np.percentile(pothole_region, 1)

    depth_diff = abs(float(d_road) - float(d_bottom))

    depth_cm = depth_diff * max_depth_cm
    depth_m = depth_cm / 100.0

    return float(depth_m)


def show_pothole_result_image(image_path, visual_reports):

    image = Image.open(image_path).convert("RGB")

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except:
        font = ImageFont.load_default()


    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for report in visual_reports:
        x1, y1, x2, y2 = report["bbox"]

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        rx = max((x2 - x1) / 2, 15)
        ry = max((y2 - y1) / 2, 15)


        overlay_draw.ellipse(
            [cx - rx, cy - ry, cx + rx, cy + ry],
            fill=(255, 0, 0, 85),
            outline=(255, 0, 0, 230),
            width=4
        )


        overlay_draw.rectangle(
            [x1, y1, x2, y2],
            outline=(255, 0, 0, 255),
            width=5
        )

    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    placed_label_boxes = []

    def is_overlap(box1, box2, margin=6):
        ax1, ay1, ax2, ay2 = box1
        bx1, by1, bx2, by2 = box2

        return not (
            ax2 + margin < bx1 or
            ax1 - margin > bx2 or
            ay2 + margin < by1 or
            ay1 - margin > by2
        )

    def clamp_label_box(x, y, w, h, img_w, img_h):
        x = max(5, min(x, img_w - w - 5))
        y = max(5, min(y, img_h - h - 5))
        return x, y

    def find_label_position(bbox, label_w, label_h):
        img_w, img_h = image.size

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        gap = 8

        candidates = [
            (x1, y1 - label_h - gap),
            (x1, y2 + gap),
            (x2 + gap, cy - label_h // 2),
            (x1 - label_w - gap, cy - label_h // 2),
            (cx - label_w // 2, y1 - label_h * 2 - gap),
            (cx - label_w // 2, y2 + label_h + gap),
        ]

        for cand_x, cand_y in candidates:
            cand_x, cand_y = clamp_label_box(cand_x, cand_y, label_w, label_h, img_w, img_h)
            cand_box = [cand_x, cand_y, cand_x + label_w, cand_y + label_h]

            if all(not is_overlap(cand_box, placed) for placed in placed_label_boxes):
                return cand_x, cand_y, cand_box

        for y in range(5, img_h - label_h - 5, label_h + 8):
            for x in range(5, img_w - label_w - 5, 30):
                cand_box = [x, y, x + label_w, y + label_h]

                if all(not is_overlap(cand_box, placed) for placed in placed_label_boxes):
                    return x, y, cand_box

        cand_x, cand_y = clamp_label_box(x1, y1 - label_h - gap, label_w, label_h, img_w, img_h)
        cand_box = [cand_x, cand_y, cand_x + label_w, cand_y + label_h]
        return cand_x, cand_y, cand_box

    visual_reports_sorted = sorted(
        visual_reports,
        key=lambda r: r["bbox"][3],
        reverse=True
    )

    for report in visual_reports_sorted:
        x1, y1, x2, y2 = report["bbox"]

        depth_cm = report["depth_m"] * 100
        damage_type = report["type"]

        label = f"{damage_type} / {depth_cm:.1f}cm"

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        pad_x = 8
        pad_y = 6

        label_w = text_w + pad_x * 2
        label_h = text_h + pad_y * 2

        label_x, label_y, label_box = find_label_position(
            [x1, y1, x2, y2],
            label_w,
            label_h
        )

        placed_label_boxes.append(label_box)


        box_cx = (x1 + x2) // 2
        box_cy = (y1 + y2) // 2
        label_cx = label_x + label_w // 2
        label_cy = label_y + label_h // 2

        draw.line(
            [label_cx, label_cy, box_cx, box_cy],
            fill=(255, 0, 0),
            width=3
        )

        draw.rectangle(
            [label_x, label_y, label_x + label_w, label_y + label_h],
            fill=(255, 0, 0)
        )


        draw.text(
            (label_x + pad_x, label_y + pad_y),
            label,
            fill=(255, 255, 255),
            font=font
        )

    display(image)


def run_pothole_detection_image_only(
    target_image=TARGET_IMAGE,
    model_dir=MODEL_DIR,
    yolo_weights_path=YOLO_WEIGHTS_PATH,
    max_depth_cm=20.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device: {device}")

    if not os.path.exists(target_image):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {target_image}")

    if not os.path.exists(yolo_weights_path):
        raise FileNotFoundError(f"YOLO 가중치 파일을 찾을 수 없습니다: {yolo_weights_path}")

    input_image = Image.open(target_image).convert("RGB")
    orig_w, orig_h = input_image.size

    # U-Net 입력 전처리
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = preprocess(input_image).unsqueeze(0).to(device)

    # U-Net 모델 로드
    model = PotholeUNet().to(device)
    model.eval()

    ensemble_pred = torch.zeros((1, 1, 256, 256), device=device)

    print(">> U-Net depth map 예측 중...")

    with torch.no_grad():
        for fold in range(1, 6):
            model_path = os.path.join(model_dir, f"best_model_fold{fold}.pth")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

            print(f"로딩 중: {model_path}")

            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

            output = model(input_tensor)
            ensemble_pred += output

    ensemble_pred = ensemble_pred / 5.0

    pred_map = ensemble_pred.squeeze().detach().cpu().numpy()
    pred_map = normalize_depth_map(pred_map)

    pred_map_resized = cv2.resize(
        pred_map,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR
    )

    # 정상 도로 기준 depth 계산
    print(">> 정상 도로 기준 depth 계산 중...")

    road_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    pts = np.array([
        [int(orig_w * 0.4), int(orig_h * 0.3)],
        [int(orig_w * 0.6), int(orig_h * 0.3)],
        [orig_w, orig_h],
        [0, orig_h]
    ], np.int32)

    cv2.fillPoly(road_mask, [pts], 255)

    roi_pixels = pred_map_resized[road_mask == 255]
    d_road = np.median(roi_pixels) if roi_pixels.size > 0 else 0.0

    print(f">> 기준 도로 depth: {d_road:.4f}")

    # YOLO 탐지
    print(">> YOLO 포트홀 탐지 중...")

    yolo_model = YOLO(yolo_weights_path)

    detections = yolo_model(
        target_image,
        conf=0.25,
        imgsz=640,
        iou=0.7,
        verbose=False
    )

    visual_reports = []

    for result in detections:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        class_names = yolo_model.names

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(orig_w, int(x2))
            y2 = min(orig_h, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = int(class_ids[idx])
            raw_name = class_names[cls_id]

            if raw_name.lower() == "pothole":
                damage_type = "pothole"
            else:
                damage_type = raw_name.lower()

            pothole_region = pred_map_resized[y1:y2, x1:x2]

            depth_m = calculate_real_depth(
                d_road=d_road,
                pothole_region=pothole_region,
                max_depth_cm=max_depth_cm
            )

            visual_reports.append({
                "type": damage_type,
                "bbox": [x1, y1, x2, y2],
                "depth_m": round(depth_m, 3)
            })

    print("\n>> 탐지 결과")
    if len(visual_reports) == 0:
        print("탐지된 포트홀이 없습니다.")
        display(input_image)
        return []

    for r in visual_reports:
        depth_cm = r["depth_m"] * 100
        print(f'{r["type"]} / {depth_cm:.1f}cm / bbox={r["bbox"]}')

    print("\n>> 결과 이미지")
    show_pothole_result_image(target_image, visual_reports)

    return visual_reports

visual_reports = run_pothole_detection_image_only()