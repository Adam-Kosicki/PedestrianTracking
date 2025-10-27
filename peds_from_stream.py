import cv2
from ultralytics import YOLO

# Define RTSP stream URL
rtsp_url = "rtsp://root:root@192.168.6.149/axis-media/media.amp?videocodec=h264&resolution=1280x720"

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # Use yolov8n.pt for the nano version, or other variants

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from stream.")
        break

    # Perform object detection
    results = model(frame)

    # Visualize results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLO Detection on RTSP", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()