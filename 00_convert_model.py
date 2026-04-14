# 00_convert_model.py
from ultralytics import YOLO
import os 
# Load a model
# model = YOLO("runs/detect/train6/weights/best.pt")  # load a custom trained model

model = YOLO("./weights/best.pt")  # load a custom trained model

# half	bool default False	Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware. Not compatible with INT8 quantization or CPU-only exports for ONNX.
# simplify	bool default True	Simplifies the model graph for ONNX exports with onnxslim, potentially improving performance and compatibility with inference engines.
# imgsz	int or tuple	640	Desired image size for the model input. Can be an integer for square images (e.g., 640 for 640×640) or a tuple (height, width) for specific dimensions.
# nms	bool default False	Adds Non-Maximum Suppression (NMS) to the exported model when supported (see Export Formats), improving detection post-processing efficiency. Not available for end2end models.

# Export the model
# model.export(format="onnx", half=True) -> best.onnx
 
# for ONNX imgsz, half, dynamic, simplify, opset, nms, batch, device 
# model.export(format="onnx", half=False, nms=True, score_threshold=0.55) 
model.export(format="onnx")


os.rename("./weights/best.onnx", "./weights/best_NIPA_Data_2025_v8_train31.onnx")
  