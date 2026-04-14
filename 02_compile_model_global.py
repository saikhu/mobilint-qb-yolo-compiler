# Step 2: Compile the model with the calibration dataset that was prepared during step 1.

from qubee import mxq_compile

# onnx_model_path = "runs/detect/train6/weights/best.onnx"   # -> yolov8
onnx_model_path = "weights/best_NIPA_Data_2025_v8_train31.onnx"
calib_data_path = "calibrated_dataset/NIPA_Data_2025_v8"

# for the best.mxq
# mxq_compile( 
#     model=onnx_model_path,
#     calib_data_path=calib_data_path,
#     quantize_method= "maxpercentile",
#     is_quant_ch=True,
#     quantize_percentile=0.999,
#     topk_ratio=0.01,
#     quant_output="ch",
#     save_path="yolov8l_g.mxq",
#     backend="onnx"
# )

mxq_compile(
    model=onnx_model_path,
    calib_data_path=calib_data_path,
    quantize_method= "kl",
    is_quant_ch=True,
    quantize_percentile=0.999,
    topk_ratio=0.01,
    quant_output="ch",
    save_path="yolov11_NIPA_Data_2025_v8.mxq",
    backend="onnx",
    yolo_decode_include=True,
    # score_threshold = 0.55,
    inference_scheme="global"
)