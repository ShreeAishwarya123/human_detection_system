import math
import cv2

# Define the brake radius (adjust as needed)
BRAKE_RADIUS = 150  # In pixels (adjust based on camera view)

# Function to calculate Euclidean distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to check if any object is within the brake radius
def check_animal_within_radius(object_positions, car_position):
    for obj in object_positions:
        distance = calculate_distance(car_position[0], car_position[1], obj[0], obj[1])
        if distance <= BRAKE_RADIUS:
            return True
    return False

# Function to draw bounding boxes and labels on the frame
def draw_detections(frame, detections):
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        label = f"ID {class_id}: {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame
