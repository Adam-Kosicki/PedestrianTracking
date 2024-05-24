import cv2
from ultralytics import YOLO


def main():
    # Initialize the YOLOv9 model
    model = YOLO('yolov9e-seg.pt')

    # Define the video source
    video_path = 'video.mp4'
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened correctly
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Read and process the video
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Perform detection and tracking
        results = model.track(frame, persist=True, conf=0.2, iou=0.5, show=True, tracker="bytetrack.yaml")

        # Get annotated frame and display it
        annotated_frame = results[0].plot()
        cv2.imshow('YOLOv9 Tracking', annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
