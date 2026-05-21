import argparse
import asyncio

import cv2
import websockets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream video frames to the realtime pothole backend")
    parser.add_argument("--source", default=0, help="Webcam index or video file path")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8000/ws/ingest", help="Backend websocket ingest URL")
    parser.add_argument("--fps", type=float, default=15.0, help="Target send fps")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality")
    return parser


async def stream_frames(ws_url: str, source: str, fps: float, quality: int) -> None:
    capture_source = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    frame_interval = 1.0 / max(1.0, fps)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(10, min(100, quality))]

    async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as websocket:
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                ok, buffer = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    continue

                await websocket.send(buffer.tobytes())

                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.2)
                    print(response)
                except asyncio.TimeoutError:
                    pass

                await asyncio.sleep(frame_interval)
        finally:
            capture.release()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(stream_frames(args.ws_url, str(args.source), args.fps, args.quality))


if __name__ == "__main__":
    main()