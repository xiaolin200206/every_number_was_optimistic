from ultralytics import YOLO
for n in ["yolo11n", "yolo11s", "yolo11m"]:
    YOLO(f"{n}.pt").export(format="onnx", imgsz=640, simplify=True, opset=12)
