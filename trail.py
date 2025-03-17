import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not detected. Check if another application is using it.")
else:
    print("✅ Camera is working!")

cap.release()
