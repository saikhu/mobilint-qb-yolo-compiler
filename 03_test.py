# 03_test.py
import os
from argparse import ArgumentParser
# from mblt_model_zoo.vision import ConTiLabYOLOv11  # 👈 custom importw
import sys
sys.path.insert(0, "mblt-model-zoo")
from mblt_model_zoo.vision import ConTiLabYOLOv11



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, default="calibration_images/hamyanghapcheon_2gonggu_04_2gonggu_20250908145500_20250908150259_5657315_frame_000465.png"
    )
   
    args = parser.parse_args()
    image_path = args.image_path
       
    # local_path="/home/mobilint/Desktop/mobilint_aries/repos/mblt-model-zoo/yolov8_sib/yolov8l.mxq",
    yolo = ConTiLabYOLOv11(
        local_path="weights/yolov11_NIPA_Data_2025_v8.mxq",
        model_type="DEFAULT",
        infer_mode="global"
    )

    input_img = yolo.preprocess(image_path)
    output = yolo(input_img)
    result = yolo.postprocess(output, conf_thres=0.6, iou_thres=0.6)

    result.plot(
        source_path=image_path,
        save_path='./results.jpg',
    )
