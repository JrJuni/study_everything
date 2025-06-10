from ultralytics.utils.benchmarks import benchmark

# 특정 포맷(예: ONNX) 벤치마크
benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, format="onnx")