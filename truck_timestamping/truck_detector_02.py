import os
import cv2
import torch
import csv
from datetime import timedelta
from ultralytics import YOLO
from tqdm import tqdm

def run_truck_detection(
    input_path,
    output_video='outputs/output_video.avi',
    output_txt='outputs/timestamps.txt',
    class_name='truck',
    check_interval=1.0,
    cooldown=5.0,
    min_conf=0.5,
    use_confirmation=False
):
    # --- Device Selection ---
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n🔧 Using device: {device.upper()}")

    # --- Load model ---
    model = YOLO('yolo12n.pt')
    model.to(device)

    # --- Prepare Video ---
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_rate = max(1, int(check_interval * fps))
    print(f"Checking every {sample_rate} frames (~{check_interval:.2f} seconds)\n")

    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (frame_width, frame_height)
    )

    timestamps = []
    confirmed_last = False  # for two-frame confirmation

    # --- Process video with progress bar ---
    for frame_number in tqdm(range(frame_count), desc="Processing video"):
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

                if label == class_name and confidence >= min_conf:
                    detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        timestamp = frame_number / fps

        should_log = (
            detected and not use_confirmation
        ) or (
            use_confirmation and detected and confirmed_last
        )

        if should_log:
            if not timestamps or (timestamp - timestamps[-1]) > cooldown:
                timestamps.append(timestamp)
                print(f"{class_name.capitalize()} confirmed at {timestamp:.2f} seconds")

        if use_confirmation:
            confirmed_last = detected

        out.write(frame)

    cap.release()
    out.release()

    # --- Save timestamps as CSV ---
    csv_path = output_txt.replace('.txt', '.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp_sec', 'timestamp_hms', 'frame_number'])
        for t in timestamps:
            frame_at_time = int(t * fps)
            hms = str(timedelta(seconds=int(t)))  # Format as hh:mm:ss
            writer.writerow([f"{t:.2f}", hms, frame_at_time])

    print(f"\nDetection complete. Timestamps saved to {csv_path}")
    return timestamps
