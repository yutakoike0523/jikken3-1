from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("yolo11n-pose.pt")  # Load your trained model

# Initialize the webcam (adjust the index if needed for your setup)
camera = cv2.VideoCapture(0)  # '0' refers to the default camera; change it if necessary

if not camera.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Set camera resolution (optional)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to exit.")

while True:
    # Read a frame from the webcam
    ret, frame = camera.read()
    if not ret:
        print("Error: Unable to read from the webcam.")
        break

    # Run YOLO inference
    results = model(frame)  # Run inference on the current frame

    # Visualize results on the frame
    annotated_frame = results[0].plot()  # Annotate the frame with YOLO results

    # Display the annotated frame
    cv2.imshow("YOLOv11 Pose Detection", annotated_frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
