import cv2
from ultralytics import YOLO
import os
import argparse
import sys

# Define RTSP stream URL (default)
rtsp_url = "rtsp://root:root@192.168.6.149/axis-media/media.amp?videocodec=h264&resolution=1280x720"

# Use same model and tracker config as PedestrianDetectorV1
weights_path = "yolov10s.pt"
tracker_cfg_path = "botsort_pedestrian.yaml"

# Load YOLOv10s weights
model = YOLO(weights_path)

# CLI to switch between RTSP and local MP4 file
parser = argparse.ArgumentParser(description="Pedestrian tracking from RTSP or MP4 using Ultralytics + BoT-SORT")
parser.add_argument("--source", choices=["rtsp", "file"], default="rtsp", help="Input source type")
parser.add_argument("--rtsp-url", default=rtsp_url, help="RTSP URL (used when --source rtsp)")
parser.add_argument("--video", help="Path to MP4 file (used when --source file)")
parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
args = parser.parse_args()

if args.source == "rtsp":
    input_source = args.rtsp_url
    window_title = "YOLO Pedestrian Tracking - RTSP"
else:
    if not args.video:
        print("Error: --video is required when --source file")
        sys.exit(1)
    input_source = args.video
    window_title = f"YOLO Pedestrian Tracking - {os.path.basename(args.video)}"

# Open the chosen source
cap = cv2.VideoCapture(input_source)

if not cap.isOpened():
    print(f"Error: Could not open source: {input_source}")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from stream.")
        break

    # Track pedestrians using BoT-SORT config
    results = model.track(
        frame,
        conf=args.conf,
        classes=[0],  # person class
        persist=True,
        tracker=tracker_cfg_path,
        verbose=False,
    )

    # Visualize results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow(window_title, annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()