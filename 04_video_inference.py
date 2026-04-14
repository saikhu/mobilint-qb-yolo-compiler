import os
import cv2
from argparse import ArgumentParser
import sys
sys.path.insert(0, "mblt-model-zoo")
from mblt_model_zoo.vision import ConTiLabYOLOv11

def run_video_inference(video_path, save_dir, model_path):
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    assert os.path.exists(model_path), f"Model not found: {model_path}"
    os.makedirs(save_dir, exist_ok=True)

    # Generate output filename automatically
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_filename = f"{video_name}_{model_name}_result.mp4"
    save_path = os.path.join(save_dir, output_filename)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # Load model
    yolo = ConTiLabYOLOv11(
        local_path=model_path,
        model_type="DEFAULT",
        infer_mode="global"
    )

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        temp_path = "/tmp/_inference_frame.jpg"
        cv2.imwrite(temp_path, frame)

        input_img = yolo.preprocess(temp_path)
        output = yolo(input_img)
        result = yolo.postprocess(output, conf_thres=0.55, iou_thres=0.55)

        annotated_frame = result.plot(source_path=temp_path, save_path=None)
        out.write(annotated_frame[:, :, ::-1])  # RGB → BGR for OpenCV

        frame_id += 1
        print(f"[{frame_id}] Processed frame")

    cap.release()
    out.release()
    print(f"\n✅ Inference complete. Saved to: {save_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--save_dir", type=str, default="test_videos/results/", help="Directory to save output video")
    parser.add_argument("--model_path", type=str, default="yolov8l.mxq",  help="Path to .mxq model")

    args = parser.parse_args()
    run_video_inference(args.video_path, args.save_dir, args.model_path)
