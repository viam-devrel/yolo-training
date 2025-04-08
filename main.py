import platform
from typing import Optional
from ultralytics import YOLO

device: Optional[str] = "cpu"
platform_str = platform.platform().lower()

if "macos" in platform_str and "arm64" in platform_str:
    device = "mps"

model = YOLO("yolov8n.pt")

results = model.train(
    data="coco8.yaml", epochs=100, imgsz=640, device="mps", patience=30
)

model.export(format="onnx", device=device)
