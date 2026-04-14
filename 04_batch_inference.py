import os
import glob
import sys
from argparse import ArgumentParser
sys.path.insert(0, "mblt-model-zoo")
from mblt_model_zoo.vision import ConTiLabYOLOv11

def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))

def run_inference_on_dir(image_dir, save_dir, model_path):
    os.makedirs(save_dir, exist_ok=True)

    # Load model once
    yolo = ConTiLabYOLOv11(
        local_path=model_path,
        model_type="DEFAULT",
        infer_mode="global"
    )

    image_paths = sorted(
        [os.path.join(image_dir, f) for f in os.listdir(image_dir) if is_image_file(f)]
    )

    if not image_paths:
        print(f"No image files found in: {image_dir}")
        return

    for img_path in image_paths:
        print(f"Inferencing on: {img_path}")

        input_img = yolo.preprocess(img_path)
        output = yolo(input_img)
        result = yolo.postprocess(output, conf_thres=0.6, iou_thres=0.6)

        save_path = os.path.join(save_dir, os.path.basename(img_path))
        result.plot(source_path=img_path, save_path=save_path)

    print(f"\n✅ Inference completed. Results saved to: {save_dir}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="../NIPA_Data_2025_v8/valid/images", help="Directory of input images")
    parser.add_argument("--save_dir", type=str, default="inference_results", help="Directory to save output")
    parser.add_argument("--model_path", type=str, default="weights/yolov11_NIPA_Data_2025_v8.mxq", help="Path to .mxq model")

    args = parser.parse_args()
    
    run_inference_on_dir(args.image_dir, args.save_dir, args.model_path)
