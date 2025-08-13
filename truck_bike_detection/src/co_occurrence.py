import os
import json
import cv2
import pandas as pd
from datetime import timedelta

def get_timestamp(frame_idx, fps):
    total_seconds = frame_idx / fps
    return str(timedelta(seconds=int(total_seconds)))

def extract_co_occurrence(
    video_path,
    detections_path,
    target_classes,
    output_csv='output/co_occurrence.csv',
    output_frame_dir='output/frames',
    save_frames=True
):
    """
    Identify frames with co-occurring classes and optionally save images.

    Args:
        video_path (str): Path to the video.
        detections_path (str): Path to JSON with detection results.
        target_classes (list): List of classes to check for co-occurrence.
        output_csv (str): Where to save the co-occurrence summary.
        output_frame_dir (str): Where to save frame images.
        save_frames (bool): Whether to save the actual frames.
    """
    os.makedirs(output_frame_dir, exist_ok=True)

    with open(detections_path, 'r') as f:
        detections = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    matched = []
    for frame_key, info in detections.items():
        classes = info.get("classes", [])
        frame_idx = int(frame_key.replace("frame_", ""))

        if all(cls in classes for cls in target_classes):
            timestamp = get_timestamp(frame_idx, fps)
            record = {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "classes": classes
            }
            matched.append(record)

            if save_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_path = os.path.join(output_frame_dir, f"frame_{frame_idx}.jpg")
                    cv2.imwrite(frame_path, frame)

    cap.release()

    df = pd.DataFrame(matched)
    df.to_csv(output_csv, index=False)
    print(f"\nCo-occurrence results saved to {output_csv} ({len(df)} matches)")
    return df
