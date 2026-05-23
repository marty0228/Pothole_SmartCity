import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = BASE_DIR / "weights" / "best.pt"
DEFAULT_DEPTH_MODEL_DIR = BASE_DIR / "best_model"

RESULT_FILE_PATH = BASE_DIR / 'result.json'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pothole-realtime")


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PotholeUNet(nn.Module):
    def __init__(self):
        super().__init__()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return torch.sigmoid(self.out(u1))


def calculate_real_depth(d_road: float, pothole_region: np.ndarray, max_depth_cm: float = 20.0) -> float:
    if pothole_region.size == 0:
        return 0.0

    d_bottom = np.percentile(pothole_region, 1)
    pixel_diff = abs(d_road - d_bottom)
    return float((pixel_diff * max_depth_cm) / 100.0)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_depth_models(depth_model_dir: Path, device: torch.device) -> list[PotholeUNet]:
    models: list[PotholeUNet] = []
    if not depth_model_dir.exists():
        logger.warning("depth model directory not found: %s", depth_model_dir)
        return models

    for fold in range(1, 6):
        weight_path = depth_model_dir / f"best_model_fold{fold}.pth"
        if not weight_path.exists():
            logger.warning("depth model missing: %s", weight_path)
            continue

        model = PotholeUNet().to(device)
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    return models


@dataclass
class InferenceResult:
    items: list[dict[str, Any]]
    frame_size: tuple[int, int]


class RealtimeDetector:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(str(WEIGHTS_PATH))
        depth_model_dir = Path(os.getenv("DEPTH_MODEL_DIR", str(DEFAULT_DEPTH_MODEL_DIR)))
        self.depth_models = _load_depth_models(depth_model_dir, self.device)

    def _prepare_depth_map(self, image_rgb: np.ndarray) -> np.ndarray | None:
        if not self.depth_models:
            return None

        image = cv2.resize(image_rgb, (256, 256))
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)

        ensemble_pred = torch.zeros((1, 1, 256, 256), device=self.device)
        with torch.no_grad():
            for model in self.depth_models:
                ensemble_pred += model(tensor)

        pred_map = (ensemble_pred / max(1, len(self.depth_models))).squeeze().detach().cpu().numpy()
        return pred_map

    def _estimate_d_road(self, pred_map: np.ndarray) -> float:
        mask = np.zeros((256, 256), dtype=np.uint8)
        polygon = np.array([
            [int(256 * 0.4), int(256 * 0.3)],
            [int(256 * 0.6), int(256 * 0.3)],
            [256, 256],
            [0, 256],
        ], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        roi_pixels = pred_map[mask == 255]
        return float(np.median(roi_pixels)) if roi_pixels.size > 0 else 0.0

    def infer_frame(self, frame_bytes: bytes) -> InferenceResult:
        decoded = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(decoded, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("frame decode failed")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pred_map = self._prepare_depth_map(frame_rgb)

        original_height, original_width = frame_rgb.shape[:2]
        pred_map_resized = cv2.resize(pred_map, (original_width, original_height)) if pred_map is not None else None
        d_road = self._estimate_d_road(pred_map) if pred_map is not None else 0.0

        detections = self.yolo(frame_rgb, conf=0.25, imgsz=640, iou=0.7, verbose=False)
        items: list[dict[str, Any]] = []
        date_str = datetime.now().strftime("%Y%m%d")
        iso_time_str = _now_iso()

        for result in detections:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            class_names_dict = self.yolo.names

            for index, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                conf = float(confs[index])
                cls_id = int(class_ids[index])
                raw_type_name = str(class_names_dict[cls_id]).lower()

                if raw_type_name == "pothole":
                    final_type_name = "pothole"
                    prefix = "ph"
                else:
                    final_type_name = "crack"
                    prefix = "cr"

                x, y = max(0, int(x1)), max(0, int(y1))
                w, h = max(1, int(x2 - x1)), max(1, int(y2 - y1))

                depth_m = 0.0
                if pred_map_resized is not None:
                    pothole_region = pred_map_resized[y:y + h, x:x + w]
                    depth_m = calculate_real_depth(d_road, pothole_region)

                pothole_id = f"{prefix}-{date_str}-{str(len(items) + 1).zfill(3)}"

                items.append({
                    "id": pothole_id,
                    "type": final_type_name,
                    "lat": 37.551302,
                    "lng": 127.075108,
                    "width_m": float(w),
                    "length_m": float(h),
                    "depth_m": round(depth_m, 3),
                    "confidence": round(conf, 2),
                    "detected_at": iso_time_str,
                })

        return InferenceResult(items=items, frame_size=(original_width, original_height))


app = FastAPI(title="Pothole SmartCity Realtime API")
detector = RealtimeDetector()
viewer_clients: set[WebSocket] = set()


async def broadcast(payload: dict[str, Any]) -> None:
    message = json.dumps(payload, ensure_ascii=False)
    dead_clients: list[WebSocket] = []

    for client in viewer_clients:
        try:
            await client.send_text(message)
        except Exception:
            dead_clients.append(client)

    for client in dead_clients:
        viewer_clients.discard(client)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/viewer")
async def viewer_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    viewer_clients.add(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        viewer_clients.discard(websocket)
    except Exception:
        viewer_clients.discard(websocket)


@app.websocket("/ws/ingest")
async def ingest_socket(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            try:
                inference = detector.infer_frame(frame_bytes)
            except Exception as exc:
                await websocket.send_text(json.dumps({"event": "error", "message": str(exc)}, ensure_ascii=False))
                continue

            if inference.items:
                existing_data = []

                if RESULT_FILE_PATH.exists() and RESULT_FILE_PATH.stat().st_size > 0:
                    try:
                        with open(RESULT_FILE_PATH, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)
                    except Exception as e:
                        logger.error(f"Json 파일을 읽기 실패 : {e}")
                        existing_data = []
                
                existing_data.extend(inference.items)

                try:
                    with open(RESULT_FILE_PATH, "w", encoding="utf-8") as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    logger.error(f"Json 파일 쓰기 실패 : {e}")

            for item in inference.items:
                payload = {"event": "detection", "item": item}
                await broadcast(payload)
                await websocket.send_text(json.dumps(payload, ensure_ascii=False))

            if not inference.items:
                await websocket.send_text(json.dumps({"event": "empty"}, ensure_ascii=False))
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ai.realtime_server:app", host="0.0.0.0", port=8000, reload=False)