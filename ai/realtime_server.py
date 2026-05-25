import base64
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

RESULT_FILE_PATH = BASE_DIR / "client_3d" / "result.json" 

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


def calculate_real_depth(
    d_road: float,
    pothole_region: np.ndarray,
    max_depth_cm: float = 20.0
) -> float:
    if pothole_region.size == 0:
        return 0.0

    d_bottom = np.percentile(pothole_region, 10)
    pixel_diff = abs(d_road - d_bottom)

    return float((pixel_diff * max_depth_cm) / 100.0)

def _to_result_json_item(item: dict[str, Any]) -> dict[str, Any]:
    """
    result.json에 저장할 형식만 남긴다.
    bbox는 2D 시각화용이므로 저장하지 않는다.
    """

    return {
        "id": item["id"],
        "type": item["type"],
        "lat": item["lat"],
        "lng": item["lng"],
        "width_m": item["width_m"],
        "length_m": item["length_m"],
        "depth_m": item["depth_m"],
        "confidence": item["confidence"],
        "detected_at": item["detected_at"],
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _boxes_overlap(box1: list[int], box2: list[int], margin: int = 6) -> bool:
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    return not (
        ax2 + margin < bx1 or
        ax1 - margin > bx2 or
        ay2 + margin < by1 or
        ay1 - margin > by2
    )


def _draw_annotated_frame(frame_bgr: np.ndarray, items: list[dict[str, Any]]) -> str:
    """
    원본 프레임 위에 bbox + pothole / 6.0cm 형태의 라벨을 그린 뒤,
    base64 JPEG 문자열로 반환한다.

    - confidence는 이미지에 표시하지 않음
    - 라벨끼리 겹치면 자동으로 피해서 배치
    - 라벨과 포트홀은 빨간 선으로 연결
    """

    output = frame_bgr.copy()
    image_h, image_w = output.shape[:2]

    placed_labels: list[list[int]] = []

    # 화면 아래쪽, 즉 가까운 포트홀부터 라벨 우선 배치
    sorted_items = sorted(
        items,
        key=lambda item: item.get("bbox", [0, 0, 0, 0])[3],
        reverse=True
    )

    for item in sorted_items:
        bbox = item.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, min(x1, image_w - 1))
        y1 = max(0, min(y1, image_h - 1))
        x2 = max(0, min(x2, image_w - 1))
        y2 = max(0, min(y2, image_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        depth_cm = float(item.get("depth_m", 0.0)) * 100.0
        damage_type = str(item.get("type", "pothole"))

        label = f"{damage_type} / {depth_cm:.1f}cm"

        # -----------------------------
        # 1. 포트홀 반투명 빨간 표시
        # -----------------------------
        overlay = output.copy()

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        rx = max(int((x2 - x1) / 2), 15)
        ry = max(int((y2 - y1) / 2), 15)

        cv2.ellipse(
            overlay,
            (cx, cy),
            (rx, ry),
            0,
            0,
            360,
            (0, 0, 255),
            -1
        )

        output = cv2.addWeighted(overlay, 0.28, output, 0.72, 0)

        # bbox
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        # -----------------------------
        # 2. 라벨 크기 계산
        # -----------------------------
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2

        padding_x = 8
        padding_y = 8

        text_size, baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        text_w, text_h = text_size

        label_w = text_w + padding_x * 2
        label_h = text_h + padding_y * 2 + baseline

        gap = 8

        candidate_positions = [
            # bbox 위
            (x1, y1 - label_h - gap),

            # bbox 아래
            (x1, y2 + gap),

            # bbox 오른쪽
            (x2 + gap, cy - label_h // 2),

            # bbox 왼쪽
            (x1 - label_w - gap, cy - label_h // 2),

            # 더 위 중앙
            (cx - label_w // 2, y1 - label_h * 2 - gap),

            # 더 아래 중앙
            (cx - label_w // 2, y2 + label_h + gap),
        ]

        selected_box: list[int] | None = None
        selected_x: int | None = None
        selected_y: int | None = None

        for cand_x, cand_y in candidate_positions:
            cand_x = max(5, min(int(cand_x), image_w - label_w - 5))
            cand_y = max(5, min(int(cand_y), image_h - label_h - 5))

            cand_box = [
                cand_x,
                cand_y,
                cand_x + label_w,
                cand_y + label_h
            ]

            if all(not _boxes_overlap(cand_box, placed) for placed in placed_labels):
                selected_box = cand_box
                selected_x = cand_x
                selected_y = cand_y
                break

        # 후보가 전부 겹치면 이미지 위쪽부터 빈 공간 탐색
        if selected_box is None:
            for scan_y in range(5, max(6, image_h - label_h - 5), label_h + 8):
                for scan_x in range(5, max(6, image_w - label_w - 5), 30):
                    cand_box = [
                        scan_x,
                        scan_y,
                        scan_x + label_w,
                        scan_y + label_h
                    ]

                    if all(not _boxes_overlap(cand_box, placed) for placed in placed_labels):
                        selected_box = cand_box
                        selected_x = scan_x
                        selected_y = scan_y
                        break

                if selected_box is not None:
                    break

        # 그래도 없으면 bbox 위쪽에 강제 배치
        if selected_box is None:
            selected_x = max(5, min(x1, image_w - label_w - 5))
            selected_y = max(5, min(y1 - label_h - gap, image_h - label_h - 5))
            selected_box = [
                selected_x,
                selected_y,
                selected_x + label_w,
                selected_y + label_h
            ]

        placed_labels.append(selected_box)

        lx1, ly1, lx2, ly2 = selected_box

        # -----------------------------
        # 3. 라벨과 bbox 연결선
        # -----------------------------
        label_cx = int((lx1 + lx2) / 2)
        label_cy = int((ly1 + ly2) / 2)

        cv2.line(
            output,
            (label_cx, label_cy),
            (cx, cy),
            (0, 0, 255),
            2
        )

        # -----------------------------
        # 4. 라벨 배경 + 텍스트
        # -----------------------------
        cv2.rectangle(
            output,
            (lx1, ly1),
            (lx2, ly2),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            output,
            label,
            (lx1 + padding_x, ly1 + padding_y + text_h),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    success, buffer = cv2.imencode(
        ".jpg",
        output,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )

    if not success:
        raise ValueError("annotated frame encode failed")

    return base64.b64encode(buffer).decode("utf-8")


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
    annotated_image: str | None = None


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

        image = (
            image - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)

        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)

        ensemble_pred = torch.zeros((1, 1, 256, 256), device=self.device)

        with torch.no_grad():
            for model in self.depth_models:
                ensemble_pred += model(tensor)

        pred_map = (
            ensemble_pred / max(1, len(self.depth_models))
        ).squeeze().detach().cpu().numpy()

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

    def infer_frame(
        self,
        frame_bytes: bytes,
        lat: float,
        lng: float,
        captured_at: str | None = None
    ) -> InferenceResult:
        decoded = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(decoded, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            raise ValueError("frame decode failed")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        pred_map = self._prepare_depth_map(frame_rgb)

        original_height, original_width = frame_rgb.shape[:2]

        pred_map_resized = (
            cv2.resize(pred_map, (original_width, original_height))
            if pred_map is not None
            else None
        )

        d_road = self._estimate_d_road(pred_map) if pred_map is not None else 0.0

        detections = self.yolo(
            frame_rgb,
            conf=0.25,
            imgsz=640,
            iou=0.7,
            verbose=False
        )

        items: list[dict[str, Any]] = []

        date_str = datetime.now().strftime("%Y%m%d")
        iso_time_str = captured_at or _now_iso()

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

                x = max(0, int(x1))
                y = max(0, int(y1))

                x_end = min(original_width, int(x2))
                y_end = min(original_height, int(y2))

                w = max(1, x_end - x)
                h = max(1, y_end - y)

                depth_m = 0.0

                if pred_map_resized is not None:
                    pothole_region = pred_map_resized[y:y + h, x:x + w]
                    depth_m = calculate_real_depth(d_road, pothole_region)

                pothole_id = f"{prefix}-{date_str}-{str(len(items) + 1).zfill(3)}"

                items.append({
                    "id": pothole_id,
                    "type": final_type_name,

                    "lat": round(float(lat), 7),
                    "lng": round(float(lng), 7),

                    # 이미지 반환용 bbox
                    "bbox": [x, y, x + w, y + h],

                    # 기존 필드 유지
                    # 다만 실제 값은 meter가 아니라 pixel 크기임
                    "width_m": float(w),
                    "length_m": float(h),

                    "depth_m": round(depth_m, 3),
                    "confidence": round(conf, 2),
                    "detected_at": iso_time_str,
                })

        annotated_image = _draw_annotated_frame(frame_bgr, items) if items else None

        return InferenceResult(
            items=items,
            frame_size=(original_width, original_height),
            annotated_image=annotated_image
        )


app = FastAPI(title="Pothole SmartCity Realtime API")
detector = RealtimeDetector()
viewer_clients: set[WebSocket] = set()


def _decode_ingest_payload(message_text: str) -> tuple[bytes, float, float, str | None]:
    payload = json.loads(message_text)

    if not isinstance(payload, dict):
        raise ValueError("ingest payload must be a JSON object")

    if payload.get("event") != "frame":
        raise ValueError("ingest payload event must be 'frame'")

    image_value = payload.get("image")

    if not isinstance(image_value, str) or not image_value:
        raise ValueError("ingest payload image is required")

    lat_value = float(payload.get("lat"))
    lng_value = float(payload.get("lng"))

    if not np.isfinite(lat_value) or not np.isfinite(lng_value):
        raise ValueError("ingest payload lat/lng must be finite numbers")

    captured_at = payload.get("captured_at")
    captured_at_text = str(captured_at) if captured_at else None

    try:
        frame_bytes = base64.b64decode(image_value, validate=True)
    except Exception as exc:
        raise ValueError("ingest payload image must be valid base64 JPEG data") from exc

    return frame_bytes, lat_value, lng_value, captured_at_text


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
            message = await websocket.receive()

            if message.get("text") is None:
                await websocket.send_text(json.dumps({
                    "event": "error",
                    "message": "send JSON text payload with image, lat, and lng"
                }, ensure_ascii=False))
                continue

            try:
                frame_bytes, lat, lng, captured_at = _decode_ingest_payload(message["text"])

                inference = detector.infer_frame(
                    frame_bytes,
                    lat=lat,
                    lng=lng,
                    captured_at=captured_at
                )

            except Exception as exc:
                await websocket.send_text(json.dumps({
                    "event": "error",
                    "message": str(exc)
                }, ensure_ascii=False))
                continue

            if inference.items:
                # =========================
                # 1. result.json 저장
                # bbox는 저장하지 않고 기존 형식만 저장
                # =========================

                existing_data = []

                if RESULT_FILE_PATH.exists() and RESULT_FILE_PATH.stat().st_size > 0:
                    try:
                        with open(RESULT_FILE_PATH, "r", encoding="utf-8") as f:
                            loaded_data = json.load(f)

                        if isinstance(loaded_data, list):
                            existing_data = loaded_data
                        else:
                            logger.warning("result.json root is not a list. reset to empty list.")
                            existing_data = []

                    except Exception as e:
                        logger.error("Json 파일을 읽기 실패 : %s", e)
                        existing_data = []

                save_items = [_to_result_json_item(item) for item in inference.items]

                existing_data.extend(save_items)

                try:
                    RESULT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

                    temp_file_path = RESULT_FILE_PATH.with_suffix(".json.tmp")

                    with open(temp_file_path, "w", encoding="utf-8") as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=4)

                    os.replace(temp_file_path, RESULT_FILE_PATH)

                    logger.info(
                        "result.json 저장 완료: %s, 추가 개수: %d",
                        RESULT_FILE_PATH,
                        len(save_items)
                    )

                except Exception as e:
                    logger.error("Json 파일 쓰기 실패 : %s", e)

                # =========================
                # 2. 라벨 그려진 이미지 반환
                # bbox가 필요하므로 inference.items 그대로 사용
                # =========================

                image_payload = {
                    "event": "annotated_frame",
                    "image": inference.annotated_image,
                    "items": inference.items,
                    "frame_size": {
                        "width": inference.frame_size[0],
                        "height": inference.frame_size[1],
                    }
                }

                await broadcast(image_payload)
                await websocket.send_text(json.dumps(image_payload, ensure_ascii=False))

                # =========================
                # 3. 기존 detection 이벤트도 유지
                # =========================

                for item in inference.items:
                    payload = {
                        "event": "detection",
                        "item": item
                    }

                    await broadcast(payload)
                    await websocket.send_text(json.dumps(payload, ensure_ascii=False))

            else:
                await websocket.send_text(json.dumps({
                    "event": "empty"
                }, ensure_ascii=False))

    except WebSocketDisconnect:
        return

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ai.realtime_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )