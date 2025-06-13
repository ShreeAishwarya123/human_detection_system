from ultralytics import YOLO
import cv2
import pygame
from utils import check_animal_within_radius, draw_detections, calculate_distance

# Initialize sound system
pygame.mixer.init()
horn_sound = pygame.mixer.Sound("horn_sound.mp3")  # Ensure this file exists

# Load YOLO model
model = YOLO("yolov8n.pt")  # Ensure this file is in your directory

# Class IDs for detection
ANIMAL_CLASS_IDS = [16, 17, 18]  # 16=bird, 17=cat, 18=dog
HUMAN_CLASS_ID = 0  # 0=person

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width, _ = frame.shape
    car_position = (frame_width // 2, frame_height - 50)  # Bottom-center

    # Run YOLO on the frame
    results = model(frame)

    detections = []
    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            confidence = float(bbox.conf[0])
            class_id = int(bbox.cls[0])

            print(f"Detected Object: Class ID {class_id}, Confidence: {confidence:.2f}")

            if class_id == HUMAN_CLASS_ID or class_id in ANIMAL_CLASS_IDS:
                detections.append((x1, y1, x2, y2, confidence, class_id))

    # Check distances & play horn
    object_positions = [((d[0] + d[2]) // 2, (d[1] + d[3]) // 2) for d in detections]
    if check_animal_within_radius(object_positions, car_position):
        print("🚨 Object detected within radius! Sounding horn...")
        horn_sound.play()

    # Draw detections
    frame = draw_detections(frame, detections)
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
