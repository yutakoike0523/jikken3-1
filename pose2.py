from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained YOLO model
model = YOLO("yolo11n-hand-object.pt")  # Trained model with "hand" and "object" classes

# Initialize the webcam
camera = cv2.VideoCapture(0)  # Default webcam; adjust index if needed

if not camera.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Function to check if a hand is holding an object
def is_holding_object(detections, threshold=0.5):
    hands = []
    objects = []

    # Separate detections into "hand" and "object" categories
    for detection in detections:
        class_name = detection["name"]
        bbox = detection["box"]
        if class_name == "hand":
            hands.append(bbox)
        elif class_name == "object":
            objects.append(bbox)

    # Check for overlap between hands and objects
    for hand in hands:
        for obj in objects:
            # Calculate IoU (Intersection over Union)
            x1 = max(hand[0], obj[0])
            y1 = max(hand[1], obj[1])
            x2 = min(hand[2], obj[2])
            y2 = min(hand[3], obj[3])

            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            hand_area = (hand[2] - hand[0]) * (hand[3] - hand[1])
            obj_area = (obj[2] - obj[0]) * (obj[3] - obj[1])
            union = hand_area + obj_area - intersection

            iou = intersection / union if union > 0 else 0

            if iou > threshold:
                return True  # Hand is holding an object

    return False

print("Press 'q' to exit.")

while True:
    # Read a frame from the webcam
    ret, frame = camera.read()
    if not ret:
        print("Error: Unable to read from the webcam.")
        break

    # Run YOLO inference
    results = model(frame)  # Run inference on the current frame
    detections = results[0].boxes.data.cpu().numpy()

    # Parse detections
    parsed_detections = []
    for det in detections:
        box = det[:4]  # Bounding box coordinates
        confidence = det[4]
        class_id = int(det[5])
        class_name = model.names[class_id]
        parsed_detections.append({
            "box": [box[0], box[1], box[2], box[3]],
            "conf": confidence,
            "name": class_name
        })

    # Check if a hand is holding an object
    holding_object = is_holding_object(parsed_detections)

    # Annotate the frame with YOLO results
    annotated_frame = results[0].plot()

    # Add status text
    status_text = "Holding Object" if holding_object else "No Object"
    color = (0, 255, 0) if holding_object else (0, 0, 255)
    cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the annotated frame
    cv2.imshow("Hand-Object Detection", annotated_frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
