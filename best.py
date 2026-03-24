import cv2
from ultralytics import YOLO

print("Loading the Helmet AI model...")
# 🔴 CRITICAL CHANGE: Replace 'best.pt' with the actual path to your custom trained model.
# If you trained it locally, it is usually located in: runs/detect/train/weights/best.pt
model = YOLO('yolov8n.pt') 

print("Starting the webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not connect to the webcam.")
    exit()

print("System Armed. Monitoring for helmets...")

while True:
    success, frame = cap.read()
    if not success:
        continue

    # 1. Let the AI analyze the frame
    # We remove the 'classes' filter because your custom model should only know 'helmet' and 'no_helmet'
    results = model(frame, verbose=False)

    helmet_detected = False

    # 2. Check the AI's findings
    # We loop through everything the AI found in the frame
    for box in results[0].boxes:
        # Get the Class ID (e.g., 0 for Helmet, 1 for No Helmet)
        class_id = int(box.cls[0])
        # Get the AI's confidence level (0.0 to 1.0)
        confidence = float(box.conf[0])

        # Assuming Class 0 is 'Helmet' in your dataset, and we want to be at least 60% sure
        if class_id == 0 and confidence > 0.78:
            helmet_detected = False
        else:
            helmet_detected = True
            break # We found a helmet, no need to check the other boxes

    # 3. The Hardware Integration Logic
    if helmet_detected:
        # Visually show access granted on the screen
        cv2.putText(frame, "ACCESS GRANTED: Ignition ON", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # TODO for Hardware Phase: 
        # Insert code here to send a HIGH signal to your Raspberry Pi GPIO pin to close the relay
    else:
        # Visually show access denied
        cv2.putText(frame, "ACCESS DENIED: Wear Helmet", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # TODO for Hardware Phase:
        # Insert code here to send a LOW signal to your Raspberry Pi GPIO pin to open the relay

    # 4. Draw the bounding boxes so you can debug what the AI sees
    annotated_frame = results[0].plot()

    # 5. Show the live feed
    cv2.imshow("Smart Car Helmet Interlock", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Shutting down system...")
        break

cap.release()
cv2.destroyAllWindows()

                