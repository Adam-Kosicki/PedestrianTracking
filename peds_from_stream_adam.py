import cv2
from ultralytics import YOLO
import os
import argparse
import sys
import json
import numpy as np
from datetime import datetime, timedelta
from collections import deque

# Define RTSP stream URL (default)
rtsp_url = "rtsp://root:root@192.168.6.149/axis-media/media.amp?videocodec=h264&resolution=1280x720"

# Use same model and tracker config as PedestrianDetectorV1
weights_path = "/home/servicer/Desktop/adam_rtsp/yolov8n.pt" # Testing Adam's code with Yolov8n
tracker_cfg_path = "/home/servicer/Desktop/adam_rtsp/botsort_pedestrian.yaml"



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
    source_type = "rtsp"
else:
    if not args.video:
        print("Error: --video is required when --source file")
        sys.exit(1)
    input_source = args.video
    window_title = f"YOLO Pedestrian Tracking - {os.path.basename(args.video)}"
    source_type = "file"

# Open the chosen source
cap = cv2.VideoCapture(input_source)

if not cap.isOpened():
    print(f"Error: Could not open source: {input_source}")
    sys.exit(1)

# Episodic max pedestrians detected (resets when scene becomes empty)
max_pedestrians_detected = 0
episode_peak = 0
prev_people_in_frame = 0

# Sliding past-hour detections based on episode peaks
events_window = deque()  # (time_metric, count) -> time_metric: float seconds (file) or datetime (rtsp)
past_hour_sum = 0

# Hourly aggregation and export
_hourly_file_path = os.path.join(os.path.dirname(__file__), f"hourly_counts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
hour_bucket_key = None  # int for file (hour index), str for rtsp (YYYY-mm-dd HH)
hour_bucket_sum = 0

# ROI loading (multiple polygons supported)
base_dir = os.path.dirname(__file__)
roi_poly_path = os.path.join(base_dir, "roi_polygon.json")
polygons = []  # list of {name: str, points: [(x,y), ...]}

if os.path.exists(roi_poly_path):
    try:
        with open(roi_poly_path, "r", encoding="utf-8") as f:
            _poly = json.load(f)
        if isinstance(_poly, dict) and isinstance(_poly.get("polygons"), list):
            for item in _poly.get("polygons", []):
                name = str(item.get("name", "ROI"))
                pts = item.get("points") or []
                if isinstance(pts, list) and len(pts) >= 3:
                    polygons.append({
                        "name": name,
                        "points": [(int(p.get("x", 0)), int(p.get("y", 0))) for p in pts],
                    })
        elif isinstance(_poly, dict) and isinstance(_poly.get("points"), list):
            name = str(_poly.get("name", "ROI"))
            pts = _poly.get("points")
            polygons.append({
                "name": name,
                "points": [(int(p.get("x", 0)), int(p.get("y", 0))) for p in pts],
            })
    except Exception:
        polygons = []

# Track unique IDs inside each ROI per episode (reset when frame empty)
unique_ids_in_roi_by_poly = {p["name"]: set() for p in polygons}

# Track path points per ID to filter stationary pedestrians
PATH_POINTS_THRESHOLD = 15
centers_history_by_id = {}

# Pending entries (by polygon) waiting to reach threshold before logging
pending_entries_by_poly = {p["name"]: {} for p in polygons}  # name -> {tid: ts_str_at_entry}

# Per-frame inside tracking for entry events
ids_inside_prev_by_poly = {p["name"]: set() for p in polygons}

# Open log file for ROI entry events
_session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(os.path.dirname(__file__), f"roi_entries_{_session_time}.txt")
try:
    _log_f = open(log_path, "a", encoding="utf-8")
    _log_f.write(f"# ROI Entry Log | session={_session_time} | source_type={source_type} | source={input_source}\n")
    if polygons:
        for p in polygons:
            _log_f.write(f"# ROI polygon name={p['name']} points={p['points']}\n")
    _log_f.flush()
except Exception:
    _log_f = None

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

    # Derive current people count and update episodic max
    res = results[0] if results else None
    if res is not None and res.boxes is not None and res.boxes.xyxy is not None:
        people_in_frame = int(res.boxes.xyxy.shape[0])
    else:
        people_in_frame = 0

    if people_in_frame > 0:
        episode_peak = max(episode_peak, people_in_frame)
        max_pedestrians_detected = episode_peak
    else:
        # Episode ended -> commit a peak event if there was one
        if prev_people_in_frame > 0 and episode_peak > 0:
            # Determine time metric for event
            try:
                if source_type == "file":
                    now_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    # Sliding window maintenance
                    events_window.append((now_sec, episode_peak))
                    past_hour_sum += episode_peak
                    # Expire events older than 1 hour
                    cutoff = now_sec - 3600.0
                    while events_window and events_window[0][0] < cutoff:
                        old_t, old_c = events_window.popleft()
                        past_hour_sum -= old_c
                    # Hourly bucket (video hour index since start)
                    current_key = int(now_sec // 3600.0)
                else:
                    now_dt = datetime.now()
                    # Sliding window maintenance
                    events_window.append((now_dt, episode_peak))
                    past_hour_sum += episode_peak
                    cutoff_dt = now_dt - timedelta(hours=1)
                    while events_window and events_window[0][0] < cutoff_dt:
                        old_t, old_c = events_window.popleft()
                        past_hour_sum -= old_c
                    # Hourly bucket (wall clock hour)
                    current_key = now_dt.strftime("%Y-%m-%d %H")
            except Exception:
                # Fallback to wall clock in case of errors
                now_dt = datetime.now()
                events_window.append((now_dt, episode_peak))
                past_hour_sum += episode_peak
                cutoff_dt = now_dt - timedelta(hours=1)
                while events_window and events_window[0][0] < cutoff_dt:
                    old_t, old_c = events_window.popleft()
                    past_hour_sum -= old_c
                current_key = now_dt.strftime("%Y-%m-%d %H")

            # Handle hourly bucket rollover and export
            try:
                if hour_bucket_key is None:
                    hour_bucket_key = current_key
                    hour_bucket_sum = 0
                if current_key != hour_bucket_key:
                    with open(_hourly_file_path, "a", encoding="utf-8") as hf:
                        hf.write(f"hour={hour_bucket_key}\tcount={hour_bucket_sum}\tsource_type={source_type}\tsource={input_source}\n")
                    hour_bucket_key = current_key
                    hour_bucket_sum = 0
                hour_bucket_sum += episode_peak
            except Exception:
                pass

        # Reset when everyone leaves the frame
        max_pedestrians_detected = 0
        episode_peak = 0
        for k in list(unique_ids_in_roi_by_poly.keys()):
            unique_ids_in_roi_by_poly[k].clear()
    prev_people_in_frame = people_in_frame

    # Visualize results on the frame
    annotated_frame = res.plot() if res is not None else frame

    # If ROIs are defined, draw and count unique tracked IDs inside per polygon
    ids_tensor = res.boxes.id if (res is not None and res.boxes is not None) else None
    boxes_xyxy = res.boxes.xyxy if (res is not None and res.boxes is not None) else None

    def _point_in_polygon(x, y, polygon):
        # ray casting algorithm
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            # Check if point is on an horizontal boundary segment
            if ((y1 > y) != (y2 > y)):
                xinters = (x2 - x1) * (y - y1) / float(y2 - y1 + 1e-9) + x1
                if x < xinters:
                    inside = not inside
        return inside

    ids_inside_now_by_poly = {p["name"]: set() for p in polygons}
    if polygons and ids_tensor is not None and boxes_xyxy is not None:
        ids_np = ids_tensor.int().cpu().numpy()
        boxes_np = boxes_xyxy.int().cpu().numpy()

        # Draw all polygons and titles
        for p in polygons:
            pts = np.array(p["points"], dtype=np.int32)
            cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
            tx, ty = p["points"][0]
            cv2.putText(annotated_frame, p["name"], (tx, max(0, ty - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        for det_idx, bbox in enumerate(boxes_np):
            tid = int(ids_np[det_idx]) if det_idx < len(ids_np) else None
            if tid is None:
                continue
            xmin, ymin, xmax, ymax = map(int, bbox)
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            # update centers history per track id
            centers_history_by_id.setdefault(tid, []).append((cx, cy))
            for p in polygons:
                if _point_in_polygon(cx, cy, p["points"]):
                    ids_inside_now_by_poly[p["name"]].add(tid)
                    if people_in_frame > 0:
                        unique_ids_in_roi_by_poly.setdefault(p["name"], set()).add(tid)

    # Log entry events per polygon: transitioned from outside->inside, but only after reaching PATH_POINTS_THRESHOLD
    if _log_f is not None and polygons:
        # capture entry timestamps for new entries
        try:
            if source_type == "file":
                ts_sec_current = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                ts_entry_str = f"{ts_sec_current:.3f}s"
            else:
                ts_entry_str = datetime.now().isoformat(timespec="seconds")
        except Exception:
            ts_entry_str = datetime.now().isoformat(timespec="seconds")

        for p in polygons:
            prev_set = ids_inside_prev_by_poly.setdefault(p["name"], set())
            now_set = ids_inside_now_by_poly.setdefault(p["name"], set())
            new_entries = now_set - prev_set
            if new_entries:
                pend = pending_entries_by_poly.setdefault(p["name"], {})
                for tid in new_entries:
                    # store the entry timestamp; we will emit once threshold is met
                    pend.setdefault(tid, ts_entry_str)

        # flush pending entries that now meet threshold
        for p in polygons:
            pend = pending_entries_by_poly.setdefault(p["name"], {})
            to_flush = []
            for tid, ts_str in pend.items():
                if len(centers_history_by_id.get(tid, [])) >= PATH_POINTS_THRESHOLD:
                    try:
                        _log_f.write(f"{ts_str}\tenter\tid={tid}\tsource_type={source_type}\tsource={input_source}\troi={p['name']}\n")
                        to_flush.append(tid)
                    except Exception:
                        pass
            for tid in to_flush:
                pend.pop(tid, None)
        try:
            _log_f.flush()
        except Exception:
            pass

    # Update prev sets for next frame
    ids_inside_prev_by_poly = ids_inside_now_by_poly

    # Overlay metrics
    cv2.putText(
        annotated_frame,
        f"Counts People in frame All: {people_in_frame}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Max pedestrians detected: {max_pedestrians_detected}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )
    # Overlay per-polygon unique counts
    y_text = 120
    for p in polygons:
        count = len(unique_ids_in_roi_by_poly.get(p["name"], set()))
        cv2.putText(
            annotated_frame,
            f"Unique in ROI ({p['name']}): {count}",
            (30, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 200, 0),
            2,
        )
        y_text += 40

    # Overlay past-hour detections (sliding window)
    cv2.putText(
        annotated_frame,
        f"Detections past hour: {int(past_hour_sum)}",
        (30, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (200, 200, 0),
        2,
    )
    y_text += 40

    # Display the annotated frame
    cv2.imshow(window_title, annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
try:
    if _log_f is not None:
        _log_f.close()
except Exception:
    pass

# Flush last hour bucket on exit
try:
    if hour_bucket_key is not None:
        with open(_hourly_file_path, "a", encoding="utf-8") as hf:
            hf.write(f"hour={hour_bucket_key}\tcount={hour_bucket_sum}\tsource_type={source_type}\tsource={input_source}\n")
except Exception:
    pass
