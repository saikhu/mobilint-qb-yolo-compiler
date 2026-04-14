import os
import cv2
from argparse import ArgumentParser
import sys
import time

sys.path.insert(0, "mblt-model-zoo")
from mblt_model_zoo.vision import ConTiLabYOLOv11


VIDEO_EXTS_DEFAULT = (".mp4", ".avi", ".mov", ".mkv")


def build_save_path(video_path, model_path, save_dir=None, same_dir=False):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    if same_dir:
        out_dir = os.path.dirname(video_path)
        output_filename = f"NPU_{model_name}_{video_name}.mp4"
    else:
        out_dir = save_dir or "test_videos/results/"
        output_filename = f"{video_name}_{model_name}_result.mp4"

    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, output_filename)


def add_fps_overlay(frame, fps, avg_fps=None, frame_id=None):
    """
    Add FPS overlay to frame
    
    Args:
        frame: Input frame (BGR format)
        fps: Current/instantaneous FPS
        avg_fps: Average FPS (optional)
        frame_id: Frame number (optional)
    
    Returns:
        Frame with FPS overlay
    """
    # Create a copy to avoid modifying original
    display_frame = frame.copy()
    
    # Configuration for text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (0, 255, 0)  # Green
    bg_color = (0, 0, 0)  # Black background
    
    # Position for text (top-left corner)
    x, y = 10, 30
    line_height = 30
    
    # Prepare text lines
    texts = []
    if fps is not None:
        texts.append(f"FPS: {fps:.1f}")
    if avg_fps is not None:
        texts.append(f"Avg FPS: {avg_fps:.1f}")
    if frame_id is not None:
        texts.append(f"Frame: {frame_id}")
    
    # Draw each line with background
    for i, text in enumerate(texts):
        text_y = y + (i * line_height)
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(
            display_frame,
            (x - 5, text_y - text_height - 5),
            (x + text_width + 5, text_y + baseline + 5),
            bg_color,
            -1  # Filled
        )
        
        # Draw text
        cv2.putText(
            display_frame,
            text,
            (x, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
    
    return display_frame


def run_video_inference(video_path, save_dir, model_path, same_dir=False, 
                       overwrite=False, show_fps=True):
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    save_path = build_save_path(
        video_path=video_path,
        model_path=model_path,
        save_dir=save_dir,
        same_dir=same_dir
    )

    if os.path.exists(save_path) and not overwrite:
        print(f"⏭️  Skipping (already exists): {save_path}")
        return

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30

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
    temp_path = "/tmp/_inference_frame.jpg"
    
    # FPS tracking variables
    start_time = time.time()
    frame_times = []
    window_size = 30  # Number of frames to average for smooth FPS

    print(f"🎬 Starting inference on: {video_path}")

    while True:
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite(temp_path, frame)

        input_img = yolo.preprocess(temp_path)
        output = yolo(input_img)
        result = yolo.postprocess(output, conf_thres=0.55, iou_thres=0.55)

        annotated_frame = result.plot(source_path=temp_path, save_path=None)
        
        # Calculate FPS
        frame_end = time.time()
        frame_time = frame_end - frame_start
        frame_times.append(frame_time)
        
        # Keep only recent frames for moving average
        if len(frame_times) > window_size:
            frame_times.pop(0)
        
        # Calculate instantaneous and average FPS
        instant_fps = 1.0 / frame_time if frame_time > 0 else 0
        avg_fps = len(frame_times) / sum(frame_times) if frame_times else 0
        
        # Add FPS overlay if requested
        if show_fps:
            annotated_frame = add_fps_overlay(
                annotated_frame,
                fps=instant_fps,
                avg_fps=avg_fps,
                frame_id=frame_id
            )
        
        # Write frame (convert RGB to BGR for OpenCV)
        out.write(annotated_frame[:, :, ::-1])

        frame_id += 1
        
        # Progress reporting
        if frame_id % 100 == 0:
            elapsed = time.time() - start_time
            overall_fps = frame_id / elapsed if elapsed > 0 else 0
            print(f"  [{frame_id}] frames | Avg FPS: {avg_fps:.1f} | Overall: {overall_fps:.1f}")

    # Final statistics
    cap.release()
    out.release()
    
    total_time = time.time() - start_time
    overall_fps = frame_id / total_time if total_time > 0 else 0
    
    print(f"✅ Saved: {save_path}")
    print(f"📊 Total frames: {frame_id} | Time: {total_time:.1f}s | Avg FPS: {overall_fps:.1f}")


def collect_videos(root_dir, exts=VIDEO_EXTS_DEFAULT):
    videos = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            lower = fn.lower()
            if lower.startswith("npu_"):
                continue
            if lower.endswith(exts):
                videos.append(os.path.join(dirpath, fn))
    videos.sort()
    return videos


def run_dir_inference(root_dir, model_path, overwrite=False, show_fps=True):
    assert os.path.isdir(root_dir), f"Root dir not found: {root_dir}"
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    videos = collect_videos(root_dir)
    if not videos:
        print(f"⚠️ No videos found under: {root_dir}")
        return

    print(f"📁 Found {len(videos)} video(s) under {root_dir}")
    for i, vp in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing: {vp}")
        run_video_inference(
            video_path=vp,
            save_dir=None,
            model_path=model_path,
            same_dir=True,
            overwrite=overwrite,
            show_fps=show_fps
        )


if __name__ == "__main__":
    parser = ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_path", type=str, help="Path to input video")
    group.add_argument("--root_dir", type=str, help="Root directory containing subfolders of videos")

    parser.add_argument("--save_dir", type=str, default="test_videos/results/",
                        help="Directory to save output video (single-video mode)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to .mxq model")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output if it already exists")
    parser.add_argument("--no-fps", dest="show_fps", action="store_false",
                        help="Disable FPS overlay on output video")

    args = parser.parse_args()

    if args.video_path:
        run_video_inference(
            video_path=args.video_path,
            save_dir=args.save_dir,
            model_path=args.model_path,
            same_dir=False,
            overwrite=args.overwrite,
            show_fps=args.show_fps
        )
    else:
        run_dir_inference(
            root_dir=args.root_dir,
            model_path=args.model_path,
            overwrite=args.overwrite,
            show_fps=args.show_fps
        )