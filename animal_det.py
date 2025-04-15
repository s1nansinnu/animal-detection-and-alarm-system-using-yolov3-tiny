import cv2
import numpy as np
import pygame
import os

# Initialize pygame mixer
pygame.mixer.init()


# Load YOLOv3 Tiny model and class labels
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get layer names and output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Function to detect animals and trigger alarm
def detect_animals(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    animal_detected = False
    detected_animal = None  # To store the detected animal's name
    detected_confidence = 0  # To store the confidence level
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ["dog", "elephant","bear","cow"]:  # Specify relevant animal classes
                animal_detected = True
                detected_animal = classes[class_id]  # Get the detected animal's class name
                detected_confidence = confidence  # Save the confidence level
                # Get coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classes[class_id]} {round(confidence * 100, 2)}%"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if animal_detected and detected_animal:
        trigger_alarm(detected_animal, detected_confidence, frame)  # Pass the animal name and confidence to the alarm function
    return frame

def play_sound(sound_file):
    try:
        full_path = os.path.join('static', sound_file)
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.music.load(full_path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Failed to play sound {sound_file}: {e}")

# Function to trigger an alarm sound
def trigger_alarm(animal, confidence, frame):
    print(f"Animal detected: {animal} ({round(confidence * 100, 2)}%)")

    # Save snapshot
    snapshot_dir = "snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)
    filename = f"{animal}_{round(confidence * 100)}.jpg"
    filepath = os.path.join(snapshot_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"Snapshot saved at: {filepath}")

    # Play corresponding sound
    if animal == "dog":
        play_sound("dog_sound.wav")
    elif animal in ["elephant", "bear"]:
        play_sound("alarm.wav")
    else:
        print(f"No specific alarm sound set for {animal}")


# Main loop to capture video and detect animals
cap = cv2.VideoCapture("dog_video.mp4")  # Use 0 for webcam, or provide a path to a video file
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = detect_animals(frame)
    cv2.imshow("Animal Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
