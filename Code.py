from ultralytics import YOLO
import cv2
import pyttsx3
import time

# Load your trained fire detection model
model = YOLO(r"D:\Research\Promotion\6\best.pt")  # 🔁 Replace with your model path

# Fire class as per your trained model
FIRE_CLASSES = ['fire']  # Update if your class name is different

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Try to open webcam
cap = cv2.VideoCapture(0)  # Use 1 or 2 if external webcam
if not cap.isOpened():
    print("❌ Failed to open webcam. Try a different index (0, 1, or 2).")
    exit()

# Set webcam resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Voice feedback cooldown
last_spoken = ""
last_time_spoken = 0
cooldown = 3  # seconds

def speak_once(message):
    global last_spoken, last_time_spoken
    current_time = time.time()
    if message != last_spoken or (current_time - last_time_spoken) > cooldown:
        engine.say(message)
        engine.runAndWait()
        last_spoken = message
        last_time_spoken = current_time

print("✅ Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received.")
        break

    results = model(frame, verbose=False)
    detected_fire = False

    for result in results:
        boxes = result.boxes
        if hasattr(boxes, 'cls'):
            for class_index in boxes.cls.tolist():
                class_name = FIRE_CLASSES[int(class_index)]
                if class_name == 'fire':
                    detected_fire = True

    # Voice feedback only
    if detected_fire:
        speak_once("Fire detected, turn on live webcam")
    else:
        speak_once("No fire detected")

    # Show webcam feed
    cv2.imshow("Live Fire Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
