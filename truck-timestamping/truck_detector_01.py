import os
import cv2
import torch
import argparse
from tqdm import tqdm
from ultralytics import YOLO
import csv
from datetime import timedelta  


# --- Device Selection ---
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"🔧 Using device: {device.upper()}")

# --- CLI Arguments ---
parser = argparse.ArgumentParser(description="Detect objects and log timestamps.")
parser.add_argument("--input", type=str, required=True, help="Path to input video (MP4).")
parser.add_argument("--output_video", type=str, default="outputs/output_video.avi", help="Path to save annotated video.")
parser.add_argument("--output_txt", type=str, default="outputs/timestamps.txt", help="Path to save timestamps.")
parser.add_argument("--class_name", type=str, default="truck", help="Object class to detect (e.g., 'truck').")
parser.add_argument("--check_interval", type=float, default=1.0,
                    help="Interval in seconds between detection checks (default: 1.0)")
parser.add_argument("--cooldown", type=float, default=5.0,
                    help="Minimum seconds between logging repeated detections (default: 5.0)")
parser.add_argument("--min_conf", type=float, default=0.5,
                    help="Minimum confidence required for detection (default: 0.5)")
args = parser.parse_args()

# --- Load model ---
model = YOLO('yolo12n.pt')
model.to(device)

# --- Prepare Video ---
cap = cv2.VideoCapture(args.input)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

sample_rate = max(1, int(args.check_interval * fps))
print(f"🔁 Checking every {sample_rate} frames (~{args.check_interval:.2f} seconds)")

os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)

out = cv2.VideoWriter(
    args.output_video,
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (frame_width, frame_height)
)

timestamps = []

confirmed_last = False  # for two-frame confirmation

# --- Process video with progress bar ---
for frame_number in tqdm(range(frame_count), desc="🔍 Processing video"):
    ret, frame = cap.read()
    if not ret:
        break

    if frame_number % sample_rate == 0:
        results = model(frame, verbose=False)[0]
    else:
        results = None

    detected = False
    if results:
        for box in results.boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            confidence = float(box.conf[0])

            if label == args.class_name and confidence >= args.min_conf:
                detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    timestamp = frame_number / fps

    # if detected and confirmed_last:
    if detected:
        if not timestamps or (timestamp - timestamps[-1]) > args.cooldown:
            timestamps.append(timestamp)
            print(f"{args.class_name.capitalize()} confirmed at {timestamp:.2f} seconds")

    # confirmed_last = detected  # update tracker

    out.write(frame)


cap.release()
out.release()

# --- Save timestamps ---
with open(args.output_txt.replace('.txt', '.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['timestamp_sec', 'timestamp_hms', 'frame_number'])
    for t in timestamps:
        frame_at_time = int(t * fps)
        hms = str(timedelta(seconds=int(t)))  # Format as hh:mm:ss
        writer.writerow([f"{t:.2f}", hms, frame_at_time])

print(f"Detection complete. Timestamps saved to {args.output_txt.replace('.txt', '.csv')}")

