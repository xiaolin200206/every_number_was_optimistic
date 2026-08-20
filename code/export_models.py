from ultralytics import YOLO

# Model sweep: all five detectors at 640x640
for n in ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]:
    YOLO(f"{n}.pt").export(format="onnx", imgsz=640, simplify=True, opset=12)

# Iso-FLOP configurations (Table II) additionally require:
YOLO("yolo11m.pt").export(format="onnx", imgsz=1088, simplify=True, opset=12)
YOLO("yolo11n.pt").export(format="onnx", imgsz=1152, simplify=True, opset=12)
