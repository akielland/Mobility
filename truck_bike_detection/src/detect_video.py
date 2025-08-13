import os
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm
import json
from datetime import timedelta

def run_detection(
    input_path,
    model_path='yolo12n.pt',
    output_json='output/detections.json',
    sample_interval=1.0,
    min_conf=0.5
):
    """
    Run object detection on a video and save per-frame detected classes.

    Args:
        input_path (str): Path to input video.
        model_path (str): Path to YOLOv12 model weights.
        output_json (str): Path to save detection results.
        sample_interval (float): Time interval in seconds between checks.
        min_conf (float): Minimum confidence threshold to include a detection.
    """
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n🔧 Using device: {device.upper()}")

    model = YOLO(model_path)
    model.to(device)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, int(sample_interval * fps))

    detections = {}
    print(f"Running detection every {sample_rate} frames (~{sample_interval:.2f} seconds)\n")

    for frame_idx in tqdm(range(total_frames), desc="Detecting"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate != 0:
            continue

        results = model(frame, verbose=False)[0]
        classes = []

        for box in results.boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            conf = float(box.conf[0])
            if conf >= min_conf:
                classes.append(label)

        if classes:
            timestamp = str(timedelta(seconds=int(frame_idx / fps)))
            detections[f"frame_{frame_idx}"] = {
                "timestamp": timestamp,
                "classes": classes
            }

    cap.release()

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(detections, f, indent=2)

    print(f"\nDetection complete. Results saved to {output_json}")
    return detections



